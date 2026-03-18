"""
Evaluation Metrics Module for BT-TPF Framework

Implements the evaluation metrics from Section 5.1:
- Accuracy (Equation 21)
- Recall (Equation 22)
- Precision (Equation 23)
- F1-Score (Equation 24)

Also includes confusion matrix visualization.

Reference: Section 5.1 of Wang et al. (2024)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute evaluation metrics as defined in Section 5.1.
    
    Equation 21: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Equation 22: Recall = TP / (FN + TP)
    Equation 23: Precision = TP / (TP + FP)
    Equation 24: F1 = 2TP / (2TP + FP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics for detailed analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary with per-class metrics
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    return report


def print_metrics(
    metrics: Dict[str, float],
    model_name: str = "Model"
) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"{'='*50}\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix as shown in Figures 10 and 11 of the paper.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with loss/metric histories
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    num_plots = len(history)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple models as shown in Figure 7 of the paper.
    
    Args:
        results: Dictionary of model results {model_name: metrics_dict}
        save_path: Optional path to save the figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (model_name, model_metrics) in enumerate(results.items()):
        values = [model_metrics[m] * 100 for m in metrics]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters.
    
    As per Table 3 and Table 4 in the paper.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get model size in KB.
    
    As per Table 3 and Table 4 in the paper.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in KB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_kb = (param_size + buffer_size) / 1024
    return size_kb


def print_model_info(
    model: torch.nn.Module,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Print model information including parameters and size.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        
    Returns:
        Dictionary with model info
    """
    num_params = count_parameters(model)
    size_kb = get_model_size(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model Size: {size_kb:.1f} KB")
    
    return {
        'parameters': num_params,
        'model_size_kb': size_kb
    }
