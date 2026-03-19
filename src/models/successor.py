"""
Successor Model (PoolFormer) for BT-TPF Framework

Implements the 3-layer PoolFormer-based Successor model.

Architecture (from Section 3.4):
- PoolFormer is a variant of Transformer where Multi-Head Self-Attention
  is replaced with a pooling layer (no trainable parameters)
- 3-layer network partitioned into 3 modules (1 PoolFormer block each)
- MLP block middle layer neurons = 1 (for minimum parameters)
- The pooling layer serves as the "token mixer"

Key advantage: Much smaller than Predecessor (~788-918 parameters vs ~13,440)

Reference: Section 3.4 of Wang et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PoolingLayer(nn.Module):
    """
    Pooling layer as token mixer (replaces Multi-Head Attention).
    
    As per paper (Section 3.4):
    "In the Poolformer coding block, the Multi-Head self-Attention layer 
    is substituted with a pooling layer that contains no trainable parameters."
    
    "The computational complexity of Self-Attention is quadratic in the number 
    of tokens, whereas the complexity of pooling is linearly proportional to 
    the sequence length."
    
    Uses average pooling to aggregate information from nearby tokens.
    
    Args:
        pool_size: Size of the pooling window (default: 3)
    """
    
    def __init__(self, pool_size: int = 3):
        super(PoolingLayer, self).__init__()
        self.pool_size = pool_size
        self.padding = pool_size // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial pooling as token mixer.
        
        As per paper: "a spatial pooling operator is used as the token mixer 
        to aggregate information from nearby tokens on average, without any 
        learnable parameters"
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Pooled tensor (batch_size, seq_len, embed_dim)
        """
        # Transpose to (B, embed_dim, seq_len) for 1D pooling
        x = x.transpose(1, 2)
        
        # Apply average pooling
        # Pad to maintain sequence length
        x = F.avg_pool1d(
            x, 
            kernel_size=self.pool_size, 
            stride=1, 
            padding=self.padding
        )
        
        # Subtract mean (as per PoolFormer paper)
        # This makes it: pool(x) - x, learning the residual
        x = x - x.mean(dim=-1, keepdim=True)
        
        # Transpose back to (B, seq_len, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class SuccessorMLP(nn.Module):
    """
    MLP block for Successor model with minimal parameters.
    
    As per paper (Section 5.2):
    "In Successor, to minimize the size and number of parameters of the 
    Successor model, the number of MLP block middle layer neurons is set to 1."
    
    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer neurons (default: 1 as per paper for minimal params)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 1,  # Paper says exactly 1 neuron for minimal parameters
        dropout: float = 0.1
    ):
        super(SuccessorMLP, self).__init__()
        
        # Hidden dimension is 1 (minimal) as per paper Section 5.2
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.Tanh()  # Tanh activation as per paper
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    PoolFormer Encoder Block.
    
    Similar structure to Transformer block but with pooling instead of attention:
        z'_l = D(Pool(LN(z_{l-1}))) + z_{l-1}
        z_l = D(MLP(LN(z'_l))) + z'_l
    
    As per paper (Section 3.4):
    "In the Poolformer coding block, the Multi-Head self-Attention layer is 
    substituted with a pooling layer that contains no trainable parameters. 
    This layer serves as the token mixer, while the remaining layers of 
    Poolformer remain consistent with Transformer."
    
    Args:
        embed_dim: Embedding dimension
        mlp_hidden_dim: MLP hidden dimension (default: 1 for minimal params as per paper)
        dropout: Dropout rate
        pool_size: Size of pooling window
    """
    
    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int = 1,  # Paper says 1 neuron for Successor
        dropout: float = 0.1,
        pool_size: int = 3
    ):
        super(PoolFormerBlock, self).__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Pooling layer (replaces Multi-Head Attention)
        # No trainable parameters!
        self.pool = PoolingLayer(pool_size)
        
        # MLP with minimal hidden dimension (1 neuron as per paper)
        self.mlp = SuccessorMLP(embed_dim, mlp_hidden_dim, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PoolFormer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # First sublayer: Pooling (instead of Attention)
        x = x + self.dropout(self.pool(self.norm1(x)))
        
        # Second sublayer: MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class SuccessorModule(nn.Module):
    """
    A module of the Successor model containing PoolFormer block(s).
    
    As per paper: "The Successor model's 3-layer network is partitioned 
    into three modules, with each module comprising one Poolformer coding block."
    
    Args:
        embed_dim: Embedding dimension
        num_blocks: Number of PoolFormer blocks in this module (default: 1)
        mlp_hidden_dim: MLP hidden dimension (default: 1 as per paper)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 1,
        mlp_hidden_dim: int = 1,  # Paper says 1 neuron for Successor
        dropout: float = 0.1
    ):
        super(SuccessorModule, self).__init__()
        
        self.blocks = nn.ModuleList([
            PoolFormerBlock(embed_dim, mlp_hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks in the module"""
        for block in self.blocks:
            x = block(x)
        return x


class Successor(nn.Module):
    """
    Successor Model (Student) based on PoolFormer.
    
    Architecture as per Section 3.4:
    - Shared Patch Embedding with Predecessor
    - Positional Encoding
    - 3-layer PoolFormer (3 modules × 1 block)
    - Classification head
    
    Key features:
    - Pooling layer instead of Multi-Head Attention (no trainable params)
    - MLP hidden dimension = 1 (minimal parameters)
    - Results in ~788-918 parameters vs ~13,440 for Predecessor
    
    Args:
        input_channels: Number of input channels (default: 1)
        input_size: Input image size (default: 6 for 6×6)
        embed_dim: Embedding dimension (default: 8)
        num_modules: Number of modules (default: 3)
        blocks_per_module: PoolFormer blocks per module (default: 1)
        mlp_hidden_dim: MLP hidden dimension (default: 1 as per paper)
        num_classes: Number of output classes
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 6,
        embed_dim: int = 8,
        num_modules: int = 3,
        blocks_per_module: int = 1,
        mlp_hidden_dim: int = 1,  # Paper says 1 neuron for Successor MLP
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super(Successor, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modules = num_modules
        
        # Patch Embedding (same as Predecessor)
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2
        )
        
        # Calculate number of patches
        self.num_patches = (input_size // 2) ** 2
        
        # Positional Encoding (learnable for Successor)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Successor Modules (3 modules, each with 1 PoolFormer block)
        # MLP hidden dim = 1 as per paper Section 5.2
        self.modules_list = nn.ModuleList([
            SuccessorModule(
                embed_dim=embed_dim,
                num_blocks=blocks_per_module,
                mlp_hidden_dim=mlp_hidden_dim,  # 1 neuron
                dropout=dropout
            )
            for _ in range(num_modules)
        ])
        
        # Final Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head with pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Successor model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Patch Embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Pass through modules
        for module in self.modules_list:
            x = module(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling and classify
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        x = self.classifier(x)
        
        return x
    
    def get_module(self, idx: int) -> SuccessorModule:
        """Get a specific module by index"""
        return self.modules_list[idx]
    
    def forward_with_intermediate(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass returning intermediate outputs from each module.
        
        Useful for knowledge distillation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (final_output, list_of_intermediate_outputs)
        """
        intermediates = []
        
        # Patch Embedding + Positional Embedding
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Pass through modules, collecting intermediate outputs
        for module in self.modules_list:
            x = module(x)
            intermediates.append(x.clone())
        
        # Final processing
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x, intermediates
    
    def forward_from_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass starting from embedded input (after patch embedding).
        
        Used in BERT-of-Theseus module replacement.
        
        Args:
            x: Embedded tensor (batch_size, num_patches, embed_dim)
            
        Returns:
            Classification logits
        """
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through modules
        for module in self.modules_list:
            x = module(x)
        
        # Final processing
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
