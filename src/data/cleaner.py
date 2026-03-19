"""Data cleaning and preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple


def handle_missing_values(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input dataframe
        method: Interpolation method ('linear', 'ffill', 'bfill')
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Interpolate missing values
    if method == 'linear':
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    
    # Fill any remaining NaN with 0 (for columns that might be all NaN)
    df_clean = df_clean.fillna(0)
    
    print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def detect_outliers(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using z-score method.
    
    Args:
        df: Input dataframe
        columns: List of columns to check
        threshold: Z-score threshold
        
    Returns:
        Boolean dataframe indicating outliers
    """
    outliers = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > threshold
    
    return outliers


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to the dataframe.
    
    Args:
        df: Input dataframe with datetime index
        
    Returns:
        Dataframe with additional time features
    """
    df_enhanced = df.copy()
    
    # Extract time components
    df_enhanced['hour'] = df_enhanced.index.hour
    df_enhanced['day_of_week'] = df_enhanced.index.dayofweek  # Monday=0, Sunday=6
    df_enhanced['day_of_month'] = df_enhanced.index.day
    df_enhanced['month'] = df_enhanced.index.month
    df_enhanced['year'] = df_enhanced.index.year
    df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)
    
    # Define peak hours (example: 7-9 AM and 6-9 PM)
    df_enhanced['is_peak_hour'] = (
        ((df_enhanced['hour'] >= 7) & (df_enhanced['hour'] <= 9)) |
        ((df_enhanced['hour'] >= 18) & (df_enhanced['hour'] <= 21))
    ).astype(int)
    
    return df_enhanced


def preprocess_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Complete preprocessing pipeline.
    
    Args:
        df: Raw dataframe
        config: Configuration dictionary
        
    Returns:
        Preprocessed dataframe
    """
    print("Starting preprocessing pipeline...")
    
    # Handle missing values
    df_clean = handle_missing_values(
        df, 
        method=config['preprocessing']['interpolation_method']
    )
    
    # Add time features
    df_enhanced = add_time_features(df_clean)
    
    print("Preprocessing complete!")
    print(f"Final shape: {df_enhanced.shape}")
    
    return df_enhanced


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed data to CSV.
    
    Args:
        df: Processed dataframe
        output_path: Output file path
    """
    df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")
