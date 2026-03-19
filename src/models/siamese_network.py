"""
Siamese Network Module for Feature Dimensionality Reduction

Implements the Siamese network architecture as described in Section 3.1 of the paper.

The Siamese network:
- Maps input data to a projection space
- Uses Euclidean distance as similarity measure (Equation 1)
- Uses Contrastive Loss for training (Equation 2)
- Consists of 3-layer MLP with shared parameters

Reference: Section 3.1 of Wang et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function for Siamese network training.
    
    Equation 2 from the paper:
        L = (1/2N) * Σ[y*d² + (1-y)*max(margin-d, 0)²]
    
    Where:
        - d: Euclidean distance between two samples
        - y: Label (1 if same category, 0 if different)
        - margin: Minimum distance threshold for dissimilar samples
    
    Args:
        margin: The margin for dissimilar pairs (default: 1.0 as per paper)
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Contrastive Loss.
        
        Args:
            embedding1: First embedding tensor (batch_size, embedding_dim)
            embedding2: Second embedding tensor (batch_size, embedding_dim)
            y: Labels tensor (batch_size,) - 1 for same class, 0 for different
            
        Returns:
            Contrastive loss value
        """
        # Equation 1: Euclidean distance
        # d(p, q) = sqrt((p1-q1)² + (p2-q2)² + ... + (pn-qn)²)
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
        
        # Equation 2: Contrastive Loss
        # L = (1/2N) * Σ[y*d² + (1-y)*max(margin-d, 0)²]
        # When y=1 (same class): loss = d² (minimize distance)
        # When y=0 (different class): loss = max(margin-d, 0)² (maximize distance up to margin)
        
        # Note: torch.mean() computes (1/N) * Σ[...], so we multiply by 0.5 for the (1/2N) factor
        loss = 0.5 * torch.mean(
            y * torch.pow(euclidean_distance, 2) +
            (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss


class SiameseNetwork(nn.Module):
    """
    Siamese Network for feature dimensionality reduction.
    
    Architecture as per Section 3.1:
    - Three-layer MLP with shared parameters
    - Input layer -> Hidden layer (5 neurons) -> Output layer
    - Uses Tanh activation
    
    The network transforms input features to a lower-dimensional embedding space
    where samples of the same category are closer together.
    
    Args:
        input_dim: Dimension of input features (78 for CIC-IDS2017, 43 for TON_IoT)
        hidden_dim: Dimension of hidden layer (default: 5 as per paper)
        output_dim: Dimension of output embedding (default: 36 to reshape to 6x6)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 5,
        output_dim: int = 36
    ):
        super(SiameseNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Shared MLP with 3 layers as per paper
        # Input layer -> Hidden layer -> Output layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh activation as per paper
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single input.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Embedding tensor (batch_size, output_dim)
        """
        return self.encoder(x)
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of inputs.
        
        Args:
            x1: First input tensor (batch_size, input_dim)
            x2: Second input tensor (batch_size, input_dim)
            
        Returns:
            Tuple of (embedding1, embedding2)
        """
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to embedding space.
        
        This is used after training to transform the dataset
        for the Predecessor/Successor models.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Embedding tensor (batch_size, output_dim)
        """
        return self.forward_one(x)
    
    def encode_and_reshape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and reshape features to image format (6x6x1).
        
        As per Section 4.2: "a reshaping operation was performed to transform
        the network traffic into a 6x6x1 feature map"
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Reshaped tensor (batch_size, 1, 6, 6)
        """
        embedding = self.encode(x)
        # Reshape to 6x6x1 (C=1, H=6, W=6)
        batch_size = embedding.shape[0]
        return embedding.view(batch_size, 1, 6, 6)
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SiameseTrainer:
    """
    Trainer class for Siamese Network.
    
    Handles:
    - Training with Contrastive Loss
    - Feature encoding after training
    """
    
    def __init__(
        self,
        model: SiameseNetwork,
        margin: float = 1.0,
        learning_rate: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = ContrastiveLoss(margin=margin)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with SiamesePairDataset
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            embedding1, embedding2 = self.model(x1, x2)
            
            # Compute loss
            loss = self.criterion(embedding1, embedding2, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def encode_dataset(self, features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Encode entire dataset using trained Siamese network.
        
        Args:
            features: Input features tensor
            batch_size: Batch size for encoding
            
        Returns:
            Encoded features tensor
        """
        self.model.eval()
        encoded_features = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                encoded = self.model.encode(batch)
                encoded_features.append(encoded.cpu())
        
        return torch.cat(encoded_features, dim=0)
