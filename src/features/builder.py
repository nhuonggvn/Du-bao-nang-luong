"""Feature engineering module for household power consumption."""

import pandas as pd
import numpy as np
from typing import Tuple


def discretize_continuous(df: pd.DataFrame, column: str, n_bins: int = 3, 
                         labels: list = None) -> pd.Series:
    """Discretize a continuous variable into bins.
    
    Args:
        df: Input dataframe
        column: Column name to discretize
        n_bins: Number of bins
        labels: Labels for bins (default: ['Low', 'Medium', 'High'])
        
    Returns:
        Discretized series
    """
    if labels is None:
        if n_bins == 3:
            labels = ['Low', 'Medium', 'High']
        else:
            labels = [f'Bin_{i}' for i in range(n_bins)]
    
    discretized = pd.cut(df[column], bins=n_bins, labels=labels)
    return discretized


def create_basket_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create transaction data for association rule mining.
    
    Args:
        df: Input dataframe with continuous variables
        config: Configuration dictionary
        
    Returns:
        Binary transaction dataframe
    """
    n_bins = config['association']['discretization_bins']
    
    # Discretize main power variables
    basket_df = pd.DataFrame(index=df.index)
    
    # Discretize Global_active_power
    if 'Global_active_power' in df.columns:
        basket_df['Power_Level'] = discretize_continuous(
            df, 'Global_active_power', n_bins
        )
    
    # Discretize Global_intensity
    if 'Global_intensity' in df.columns:
        basket_df['Intensity_Level'] = discretize_continuous(
            df, 'Global_intensity', n_bins
        )
    
    # Discretize Sub_metering
    for i in [1, 2, 3]:
        col_name = f'Sub_metering_{i}'
        if col_name in df.columns:
            basket_df[f'SubMeter{i}_Level'] = discretize_continuous(
                df, col_name, n_bins
            )
    
    # Add time-based categorical features
    if 'hour' in df.columns:
        basket_df['Hour_Period'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
    
    if 'is_weekend' in df.columns:
        basket_df['Day_Type'] = df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    
    # Convert to one-hot encoding for association rules
    basket_encoded = pd.get_dummies(basket_df, prefix_sep='_')
    
    return basket_encoded


def create_daily_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily consumption profiles for clustering.
    
    Args:
        df: Input dataframe with datetime index (can be minute or hourly data)
        
    Returns:
        Dataframe with one row per day and 24 hourly features
    """
    # Resample to hourly to ensure 24 points per day
    df_hourly = df[['Global_active_power']].resample('H').mean().dropna()
    
    # Add time features
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['day_of_week'] = df_hourly.index.dayofweek
    df_hourly['is_weekend'] = (df_hourly['day_of_week'] >= 5).astype(int)
    df_hourly['date'] = df_hourly.index.date
    
    # Pivot to create 24-hour profiles
    profiles = []
    
    for date in df_hourly['date'].unique():
        day_data = df_hourly[df_hourly['date'] == date]
        
        if len(day_data) == 24:  # Only consider complete days
            # Create features: 24 hourly values for Global_active_power
            hourly_power = day_data.set_index('hour')['Global_active_power'].to_dict()
            
            profile = {
                'date': date,
                'day_of_week': day_data['day_of_week'].iloc[0],
                'is_weekend': day_data['is_weekend'].iloc[0],
                'total_consumption': day_data['Global_active_power'].sum(),
                'mean_consumption': day_data['Global_active_power'].mean(),
                'std_consumption': day_data['Global_active_power'].std(),
                'max_consumption': day_data['Global_active_power'].max(),
                'min_consumption': day_data['Global_active_power'].min(),
            }
            
            # Add hourly values
            for hour in range(24):
                profile[f'hour_{hour}'] = hourly_power.get(hour, 0)
            
            profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    if not profiles_df.empty and 'date' in profiles_df.columns:
        profiles_df = profiles_df.set_index('date')
    
    print(f"Created {len(profiles_df)} daily profiles")
    
    return profiles_df


def extract_time_series_features(df: pd.DataFrame, target_col: str = 'Global_active_power',
                                 lag_hours: int = 24) -> pd.DataFrame:
    """Extract lag features for time series forecasting.
    
    Args:
        df: Input dataframe
        target_col: Target column for forecasting
        lag_hours: Number of lag hours to include
        
    Returns:
        Dataframe with lag features
    """
    df_ts = df[[target_col]].copy()
    
    # Add lag features
    for i in range(1, lag_hours + 1):
        df_ts[f'lag_{i}'] = df_ts[target_col].shift(i)
    
    # Add rolling statistics
    for window in [6, 12, 24]:  # 6h, 12h, 24h windows
        df_ts[f'rolling_mean_{window}h'] = df_ts[target_col].rolling(window).mean()
        df_ts[f'rolling_std_{window}h'] = df_ts[target_col].rolling(window).std()
    
    # Drop rows with NaN (due to lagging)
    df_ts = df_ts.dropna()
    
    return df_ts
