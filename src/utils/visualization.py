"""
Visualization Module for BT-TPF Framework

Implements t-SNE visualization as shown in Figures 8 and 9 of the paper.

Reference: Section 6.1 of Wang et al. (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple
import torch


def visualize_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str = "t-SNE Visualization",
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    num_samples: Optional[int] = 20000,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize features using t-SNE as shown in Figures 8 and 9.
    
    As per Section 6.1:
    "We randomly selected 20,000 network traffic samples and subjected 
    them to processing with the BT-TPF model. We then used the t-SNE 
    algorithm to visualize the resulting outputs."
    
    Args:
        features: Feature array (N, D)
        labels: Label array (N,)
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
        num_samples: Number of samples to visualize (default: 20000 as per paper)
        save_path: Optional path to save the figure
    """
    # Sample if needed
    if num_samples and len(features) > num_samples:
        indices = np.random.choice(len(features), num_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Apply t-SNE
    print(f"Applying t-SNE to {len(features)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=class_name,
            alpha=0.6,
            s=10
        )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")
    
    plt.show()


def visualize_comparison(
    original_features: np.ndarray,
    processed_features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (16, 6),
    num_samples: Optional[int] = 20000,
    save_path: Optional[str] = None
) -> None:
    """
    Compare original and BT-TPF processed features using t-SNE.
    
    Creates a side-by-side comparison as in Figures 8 and 9 of the paper.
    
    Args:
        original_features: Original feature array before BT-TPF
        processed_features: Features after BT-TPF processing
        labels: Label array
        class_names: List of class names
        figsize: Figure size
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    # Sample if needed
    if num_samples and len(original_features) > num_samples:
        indices = np.random.choice(len(original_features), num_samples, replace=False)
        original_features = original_features[indices]
        processed_features = processed_features[indices]
        labels = labels[indices]
    
    # Apply t-SNE to both
    print("Applying t-SNE to original features...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    original_2d = tsne.fit_transform(original_features)
    
    print("Applying t-SNE to processed features...")
    processed_2d = tsne.fit_transform(processed_features)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    # Plot original
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        axes[0].scatter(
            original_2d[mask, 0],
            original_2d[mask, 1],
            c=[colors[idx]],
            label=class_name,
            alpha=0.6,
            s=10
        )
    axes[0].set_title('(a) Original Data', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot processed
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        axes[1].scatter(
            processed_2d[mask, 0],
            processed_2d[mask, 1],
            c=[colors[idx]],
            label=class_name,
            alpha=0.6,
            s=10
        )
    axes[1].set_title('(b) BT-TPF Processed Data', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to {save_path}")
    
    plt.show()


def extract_model_features(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from model's penultimate layer for visualization.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device for computation
        
    Returns:
        Tuple of (features, labels)
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Hook to extract features
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook on the layer before classifier
    if hasattr(model, 'norm'):
        model.norm.register_forward_hook(get_activation('features'))
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            _ = model(data)
            
            if 'features' in activation:
                feat = activation['features']
                if feat.dim() > 2:
                    feat = feat.mean(dim=1)  # Global average pooling
                features_list.append(feat.cpu().numpy())
            
            labels_list.append(target.numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels
