"""
Predecessor Model (Vision Transformer) for BT-TPF Framework

Implements the 9-layer Vision Transformer (ViT) based Predecessor model.

Architecture:
- Patch Embedding (Equation 14)
- Positional Encoding (Equations 15-17)
- 9 Transformer Encoder blocks divided into 3 modules (3 blocks each)
- Multi-Head Self-Attention (Equation 18)
- MLP blocks (Equation 19)

Reference: Section 3.4 of Wang et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer using convolution.
    
    Equation 14 from the paper:
        Xp = [x1; x2; ...; xN], N = HW/P²
    
    Implements patch embedding that divides input feature map into patches.
    
    Args:
        input_channels: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 8 as per paper)
        patch_size: Size of each patch (default: 2 as per paper)
        stride: Stride for convolution (default: 2 as per paper)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 8,
        patch_size: int = 2,
        stride: int = 2
    ):
        super(PatchEmbedding, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Convolution layer for patch embedding
        # As per paper: "convolution step is set to 2, convolution kernel size is set to 2,
        # and the number of convolution kernels is set to 8"
        self.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch embedding.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings (batch_size, num_patches, embed_dim)
        """
        # x: (B, C, H, W) -> (B, embed_dim, H', W')
        x = self.projection(x)
        
        # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, embed_dim, N)
        batch_size, embed_dim, h, w = x.shape
        x = x.flatten(2)  # (B, embed_dim, N)
        
        # Transpose to (B, N, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using Sine and Cosine functions.
    
    Equations 15-17 from the paper:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))      (Eq. 15)
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    (Eq. 16)
        z0 = Xp + PE                                      (Eq. 17)
    
    Args:
        embed_dim: Embedding dimension
        max_len: Maximum sequence length (default: 100)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 100,
        dropout: float = 0.1
    ):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Equation 15: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Equation 16: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension: (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Equation 17: z0 = Xp + PE
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention layer.
    
    As per paper: "the number of heads in the Multi-Head Attention layer is set to 2"
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (default: 2)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Attention output (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """
    MLP block for Transformer Encoder.
    
    As per paper (Section 5.2): "the number of neurons in the MLP block middle layer 
    is set to 4 times the length of the token sequence (patches)"
    
    For 6×6 input with patch_size=2, stride=2: num_patches = (6/2)² = 9
    So hidden_dim = 4 × 9 = 36
    
    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension (should be 4 × num_patches, NOT 4 × embed_dim)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 36,  # 4 × num_patches = 4 × 9 = 36 as per paper
        dropout: float = 0.1
    ):
        super(MLP, self).__init__()
        
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


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.
    
    Implements Equations 18-19 from the paper:
        z'_l = D(MHA(LN(z_{l-1}))) + z_{l-1}    (Eq. 18)
        z_l = D(MLP(LN(z'_l))) + z'_l           (Eq. 19)
    
    Where:
        - LN: Layer Normalization
        - MHA: Multi-Head Self-Attention
        - MLP: Multi-Layer Perceptron
        - D: Dropout
        
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_hidden_dim: MLP hidden dimension (4 × num_patches as per paper)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        mlp_hidden_dim: int = 36,  # 4 × num_patches = 4 × 9 = 36
        dropout: float = 0.1
    ):
        super(TransformerEncoderBlock, self).__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Self-Attention
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # MLP with hidden_dim = 4 × num_patches (NOT 4 × embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer block.
        
        Equation 18: z'_l = D(MHA(LN(z_{l-1}))) + z_{l-1}
        Equation 19: z_l = D(MLP(LN(z'_l))) + z'_l
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Equation 18: First sublayer (Attention)
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # Equation 19: Second sublayer (MLP)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class PredecessorModule(nn.Module):
    """
    A module of the Predecessor model containing multiple Transformer blocks.
    
    As per paper: "The Predecessor model's 9-layer network is partitioned 
    into three modules, with each module comprising three Transformer coding blocks."
    
    Args:
        embed_dim: Embedding dimension
        num_blocks: Number of Transformer blocks in this module (default: 3)
        num_heads: Number of attention heads
        mlp_hidden_dim: MLP hidden dimension (4 × num_patches = 36 as per paper)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 3,
        num_heads: int = 2,
        mlp_hidden_dim: int = 36,  # 4 × num_patches = 4 × 9 = 36 (NOT 4 × embed_dim!)
        dropout: float = 0.1
    ):
        super(PredecessorModule, self).__init__()
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks in the module"""
        for block in self.blocks:
            x = block(x)
        return x


class Predecessor(nn.Module):
    """
    Predecessor Model (Teacher) based on Vision Transformer.
    
    Architecture as per Section 3.4:
    - Patch Embedding
    - Positional Encoding
    - 9-layer Transformer Encoder (3 modules × 3 blocks)
    - Classification head
    
    Hyperparameters from Section 5.2:
    - Patch embedding: stride=2, kernel_size=2, num_kernels=8
    - Multi-Head Attention: num_heads=2
    - MLP middle layer: 4× token sequence length (num_patches)
    
    Args:
        input_channels: Number of input channels (default: 1)
        input_size: Input image size (default: 6 for 6×6)
        embed_dim: Embedding dimension (default: 8)
        num_modules: Number of modules (default: 3)
        blocks_per_module: Transformer blocks per module (default: 3)
        num_heads: Number of attention heads (default: 2)
        mlp_ratio: Ratio for MLP hidden dim calculation (default: 4)
        num_classes: Number of output classes
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 6,
        embed_dim: int = 8,
        num_modules: int = 3,
        blocks_per_module: int = 3,
        num_heads: int = 2,
        mlp_ratio: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super(Predecessor, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modules = num_modules
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_size=2,
            stride=2
        )
        
        # Calculate number of patches
        # For 6×6 input with patch_size=2, stride=2: (6/2) × (6/2) = 9 patches
        self.num_patches = (input_size // 2) ** 2
        
        # MLP hidden dimension = 4 × num_patches (NOT 4 × embed_dim!)
        # As per Section 5.2: "the number of neurons in the MLP block middle layer 
        # is set to 4 times the length of the token sequence (patches)"
        mlp_hidden_dim = mlp_ratio * self.num_patches  # 4 × 9 = 36
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=self.num_patches + 1,
            dropout=dropout
        )
        
        # Predecessor Modules (3 modules, each with 3 Transformer blocks)
        self.modules_list = nn.ModuleList([
            PredecessorModule(
                embed_dim=embed_dim,
                num_blocks=blocks_per_module,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,  # 36, not 32!
                dropout=dropout
            )
            for _ in range(num_modules)
        ])
        
        # Final Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.classifier = nn.Linear(embed_dim * self.num_patches, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Predecessor model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add Positional Encoding
        x = self.pos_encoding(x)
        
        # Pass through modules
        for module in self.modules_list:
            x = module(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Flatten and classify
        x = x.flatten(1)  # (B, num_patches × embed_dim)
        x = self.classifier(x)
        
        return x
    
    def get_module(self, idx: int) -> PredecessorModule:
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
        
        # Patch Embedding + Positional Encoding
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        
        # Pass through modules, collecting intermediate outputs
        for module in self.modules_list:
            x = module(x)
            intermediates.append(x.clone())
        
        # Final processing
        x = self.norm(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x, intermediates
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
