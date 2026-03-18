"""
BT-TPF: A Lightweight IoT Intrusion Detection Model
Based on Improved BERT-of-Theseus

This is the main execution script for the BT-TPF framework.

Usage:
    python main.py --dataset cicids2017 --data_path /path/to/data.csv
    python main.py --dataset toniot --data_path /path/to/data.csv
    python main.py --demo  # Run with synthetic data for testing

Reference: Wang et al. (2024) - Expert Systems With Applications 238 (2024) 122045

Authors' Implementation following the paper specifications:
- Siamese Network for feature dimensionality reduction (Section 3.1)
- Vision Transformer (ViT) based Predecessor model (Section 3.4)
- PoolFormer based Successor model (Section 3.4)
- Improved BERT-of-Theseus knowledge distillation (Sections 3.2-3.3)
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BTPTFConfig, get_cicids2017_config, get_toniot_config
from src.data.preprocessing import DataPreprocessor
from src.data.dataset_loader import load_cicids2017, load_toniot, create_synthetic_dataset
from src.trainer import BTPTFTrainer
from src.utils.metrics import (
    plot_confusion_matrix, 
    plot_training_history,
    compare_models,
    print_model_info
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='BT-TPF: Lightweight IoT Intrusion Detection Model'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cicids2017', 'toniot', 'synthetic'],
        default='synthetic',
        help='Dataset to use (default: synthetic for demo)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with synthetic data'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of pre-training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size (default: 1024 as per paper)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001 as per paper)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda if available)'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default='models/bt_tpf_model.pth',
        help='Path to save trained models'
    )
    
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plotting (for headless environments)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    print("="*70)
    print("BT-TPF: A Lightweight IoT Intrusion Detection Model")
    print("Based on Improved BERT-of-Theseus")
    print("="*70)
    print("\nReference: Wang et al. (2024)")
    print("Expert Systems With Applications 238 (2024) 122045")
    print("="*70)
    
    # Load configuration based on dataset
    if args.dataset == 'cicids2017':
        config = get_cicids2017_config()
        num_classes = 5
    elif args.dataset == 'toniot':
        config = get_toniot_config()
        num_classes = 10
    else:
        config = BTPTFConfig()
        num_classes = 5
    
    # Override config with command line arguments
    config.device = args.device
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    
    if args.epochs:
        config.training.pretrain_epochs = args.epochs
    
    # Load data
    print("\n" + "="*60)
    print("Step 1: Data Loading and Pre-processing")
    print("="*60)
    
    if args.demo or args.dataset == 'synthetic':
        print("\nRunning demo with synthetic data...")
        train_df, test_df = create_synthetic_dataset(
            num_samples=10000,
            num_features=78,
            num_classes=num_classes
        )
    elif args.dataset == 'cicids2017':
        if args.data_path is None:
            print("\nError: Please provide --data_path for CIC-IDS2017 dataset")
            print("Expected: Wednesday-workingHours.csv from CIC-IDS2017")
            sys.exit(1)
        train_df, test_df = load_cicids2017(args.data_path)
    elif args.dataset == 'toniot':
        if args.data_path is None:
            print("\nError: Please provide --data_path for TON_IoT dataset")
            print("Expected: Train_Test_Network.csv from TON_IoT")
            sys.exit(1)
        train_df, test_df = load_toniot(args.data_path)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    train_features, train_labels = preprocessor.fit_transform(train_df)
    test_features, test_labels = preprocessor.transform(test_df)
    
    print(f"\nPreprocessed data shapes:")
    print(f"  Train features: {train_features.shape}")
    print(f"  Train labels: {train_labels.shape}")
    print(f"  Test features: {test_features.shape}")
    print(f"  Test labels: {test_labels.shape}")
    print(f"  Number of classes: {preprocessor.num_classes}")
    print(f"  Class names: {preprocessor.class_names}")
    
    # Initialize trainer
    trainer = BTPTFTrainer(config)
    
    # Run full training pipeline
    results = trainer.train_full_pipeline(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        num_classes=preprocessor.num_classes
    )
    
    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print("\nModel Comparison (as per Tables 3-4 in the paper):")
    print("-"*50)
    
    predecessor_info = print_model_info(trainer.predecessor, "Predecessor (Teacher)")
    successor_info = print_model_info(trainer.successor, "Successor (BT-TPF)")
    
    print(f"\nParameter reduction: {(1 - successor_info['parameters']/predecessor_info['parameters'])*100:.1f}%")
    print(f"Size reduction: {(1 - successor_info['model_size_kb']/predecessor_info['model_size_kb'])*100:.1f}%")
    
    # Save models
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    trainer.save_models(args.save_path)
    
    # Plotting (if enabled)
    if not args.no_plots:
        try:
            # Plot training history
            plot_training_history(
                results['history'],
                title="BT-TPF Training History",
                save_path="training_history.png"
            )
            
            # Plot confusion matrix
            plot_confusion_matrix(
                results['results']['labels'],
                results['results']['predictions'],
                class_names=preprocessor.class_names,
                title="BT-TPF Confusion Matrix",
                save_path="confusion_matrix.png"
            )
            
        except Exception as e:
            print(f"\nWarning: Could not create plots: {e}")
            print("Continuing without plots...")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
