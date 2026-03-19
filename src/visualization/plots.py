"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def setup_plot_style():
    """Set up consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_time_series(df: pd.DataFrame, column: str, title: str = None, 
                     figsize: tuple = (15, 5)):
    """Plot time series data.
    
    Args:
        df: Dataframe with datetime index
        column: Column to plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[column], linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    ax.set_title(title or f'{column} Over Time')
    plt.tight_layout()
    return fig


def plot_daily_pattern(df: pd.DataFrame, column: str = 'Global_active_power'):
    """Plot average daily consumption pattern.
    
    Args:
        df: Dataframe with hour column
        column: Column to plot
    """
    if 'hour' not in df.columns:
        print("Error: 'hour' column not found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_avg = df.groupby('hour')[column].mean()
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(f'Average {column}')
    ax.set_title(f'Average Daily Pattern - {column}')
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cluster_visualization(X_pca: np.ndarray, labels: np.ndarray, 
                               cluster_names: dict = None):
    """Visualize clusters in 2D PCA space.
    
    Args:
        X_pca: PCA-transformed data (2D)
        labels: Cluster labels
        cluster_names: Optional cluster names
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.6,
        edgecolors='k',
        linewidths=0.5
    )
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Cluster Visualization (PCA)')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')
    
    # Add cluster names as legend if provided
    if cluster_names:
        for cluster_id, name in cluster_names.items():
            ax.scatter([], [], c=[plt.cm.viridis(cluster_id/max(labels))], 
                      label=f'Cluster {cluster_id}: {name}',
                      edgecolors='k', linewidths=0.5)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_forecast_results(test: pd.Series, predictions: dict, 
                          title: str = "Forecast Comparison"):
    """Plot actual vs predicted values.
    
    Args:
        test: Test series
        predictions: Dictionary of {model_name: predictions}
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot actual
    ax.plot(test.index, test.values, label='Actual', linewidth=2, alpha=0.7)
    
    # Plot predictions
    for model_name, preds in predictions.items():
        ax.plot(test.index, preds, label=model_name, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Global Active Power')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
