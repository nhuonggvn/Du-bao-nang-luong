"""Data loading module for household power consumption dataset."""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/params.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(file_path: str, config: dict) -> pd.DataFrame:
    """Load raw household power consumption data.
    
    Args:
        file_path: Path to raw data file
        config: Configuration dictionary
        
    Returns:
        Raw dataframe with datetime index
    """
    # Load data
    df = pd.read_csv(
        file_path,
        sep=';',
        low_memory=False,
        na_values=[config['preprocessing']['missing_value_symbol']]
    )
    
    # Combine Date and Time columns into datetime index
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )
    
    # Drop original Date and Time columns
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Sort by index
    df = df.sort_index()
    
    # Convert columns to numeric
    numeric_cols = df.columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df)} records from {file_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'n_records': len(df),
        'n_features': len(df.columns),
        'date_range': (df.index.min(), df.index.max()),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'basic_stats': df.describe().to_dict()
    }
    return info
