"""
Dataset Loader Module for BT-TPF Framework

Handles loading of:
- CIC-IDS2017 dataset
- TON_IoT dataset

Reference: Section 4.1 of Wang et al. (2024)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


def load_cicids2017(
    data_path: str,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CIC-IDS2017 dataset.
    
    As per the paper:
    - Uses Wednesday-workingHours.csv
    - Removes Heartbleed attacks (only 11 samples)
    - 75% training, 25% testing split
    
    Dataset contains:
    - 78 features
    - 5 classes: Benign, GoldenEye, Hulk, Slowhttptest, Slowloris
    
    Args:
        data_path: Path to the CSV file
        test_size: Test set proportion (default: 0.25)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Rename label column if needed
    if ' Label' in df.columns:
        df = df.rename(columns={' Label': 'Label'})
    
    # Remove Heartbleed attacks (only 11 samples as per paper)
    if 'Heartbleed' in df['Label'].values:
        df = df[df['Label'] != 'Heartbleed']
    
    # Remove NaN, Infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Split data (75% train, 25% test as per paper)
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['Label']
    )
    
    print(f"CIC-IDS2017 Dataset loaded:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Testing samples: {len(test_df)}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Classes: {df['Label'].nunique()}")
    print(f"  Class distribution:\n{df['Label'].value_counts()}")
    
    return train_df, test_df


def load_toniot(
    data_path: str,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TON_IoT dataset.
    
    As per the paper:
    - Uses Train_Test_Network dataset
    - 43 features
    - 10 classes: Normal, Backdoor, DDoS, DoS, Injection, MITM, 
                  Password, Ransomware, Scanning, XSS
    
    Args:
        data_path: Path to the CSV file
        test_size: Test set proportion (default: 0.25)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Standardize label column name
    label_col = None
    for col in ['Label', 'label', 'type', 'attack_type', 'Attack']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col and label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
    
    # Remove NaN, Infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Replace '-' with 0
    df = df.replace('-', 0)
    
    # Split data (75% train, 25% test as per paper)
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['Label']
    )
    
    print(f"TON_IoT Dataset loaded:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Testing samples: {len(test_df)}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Classes: {df['Label'].nunique()}")
    print(f"  Class distribution:\n{df['Label'].value_counts()}")
    
    return train_df, test_df


def create_synthetic_dataset(
    num_samples: int = 10000,
    num_features: int = 78,
    num_classes: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a synthetic dataset for testing purposes.
    
    Useful when actual datasets are not available.
    
    Args:
        num_samples: Total number of samples
        num_features: Number of features
        num_classes: Number of classes
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    
    # Generate labels with imbalanced distribution (similar to real data)
    class_weights = np.random.dirichlet(np.ones(num_classes) * 0.5)
    y = np.random.choice(num_classes, size=num_samples, p=class_weights)
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(num_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    
    # Add class labels
    class_names = ['Benign', 'Attack_Type_1', 'Attack_Type_2', 'Attack_Type_3', 'Attack_Type_4'][:num_classes]
    df['Label'] = [class_names[i] for i in y]
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=0.25, 
        random_state=random_state,
        stratify=df['Label']
    )
    
    print(f"Synthetic Dataset created:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Testing samples: {len(test_df)}")
    print(f"  Features: {num_features}")
    print(f"  Classes: {num_classes}")
    
    return train_df, test_df
