"""Clustering module for household consumption profiles."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from typing import Tuple, Dict


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize features.
    
    Args:
        X: Feature dataframe
        
    Returns:
        Tuple of (scaled_data, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def perform_kmeans(X: pd.DataFrame, config: dict) -> Tuple[np.ndarray, KMeans, dict]:
    """Perform K-Means clustering.
    
    Args:
        X: Feature dataframe
        config: Configuration dictionary
        
    Returns:
        Tuple of (labels, model, metrics)
    """
    n_clusters = config['clustering']['n_clusters']
    random_state = config['clustering']['random_state']
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Fit K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'inertia': kmeans.inertia_
    }
    
    print(f"K-Means Clustering Results:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    return labels, kmeans, metrics


def profile_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Generate cluster profiles.
    
    Args:
        X: Feature dataframe
        labels: Cluster labels
        
    Returns:
        Cluster profile dataframe
    """
    X_with_labels = X.copy()
    X_with_labels['cluster'] = labels
    
    # Calculate statistics for each cluster
    profiles = X_with_labels.groupby('cluster').agg(['mean', 'std', 'min', 'max', 'count'])
    
    # Add cluster sizes
    cluster_sizes = pd.DataFrame(X_with_labels['cluster'].value_counts()).T
    cluster_sizes.index = ['size']
    
    return profiles


def find_optimal_k(X: pd.DataFrame, k_range: range = range(2, 11),
                   random_state: int = 42) -> Dict[str, list]:
    """Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        X: Feature dataframe
        k_range: Range of k values to test
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of metrics (inertias, silhouettes, davies) for all k setup
    """
    X_scaled, _ = scale_features(X)
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette)
        
        print(f"k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette:.3f}")
    
    results = {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouettes,
    }
    
    return results


def reduce_dimensions_pca(X: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """Reduce dimensions using PCA for visualization.
    
    Args:
        X: Feature dataframe
        n_components: Number of components
        
    Returns:
        Tuple of (transformed_data, pca_model)
    """
    X_scaled, _ = scale_features(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    print(f"PCA: Explained variance = {explained_var.sum():.2%}")
    
    return X_pca, pca


def name_clusters(profiles: pd.DataFrame, labels: np.ndarray, 
                 df_original: pd.DataFrame) -> Dict[int, str]:
    """Assign meaningful names to clusters based on profiles.
    
    Args:
        profiles: Cluster profile dataframe
        labels: Cluster labels
        df_original: Original dataframe with cluster assignments
        
    Returns:
        Dictionary mapping cluster number to name
    """
    cluster_names = {}
    
    df_with_clusters = df_original.copy()
    df_with_clusters['cluster'] = labels
    
    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Analyze characteristics
        mean_consumption = cluster_data['total_consumption'].mean()
        mean_weekend = cluster_data['is_weekend'].mean()
        peak_hour = cluster_data[[f'hour_{i}' for i in range(24)]].mean().idxmax()
        
        # Generate name based on characteristics
        if mean_consumption > df_original['total_consumption'].quantile(0.75):
            consumption_level = "High"
        elif mean_consumption < df_original['total_consumption'].quantile(0.25):
            consumption_level = "Low"
        else:
            consumption_level = "Medium"
        
        day_type = "Weekend" if mean_weekend > 0.5 else "Weekday"
        peak_hour_num = int(peak_hour.split('_')[1])
        
        if peak_hour_num < 6:
            time_pattern = "Night"
        elif peak_hour_num < 12:
            time_pattern = "Morning"
        elif peak_hour_num < 18:
            time_pattern = "Afternoon"
        else:
            time_pattern = "Evening"
        
        cluster_names[cluster_id] = f"{consumption_level}-{day_type}-{time_pattern}"
    
    return cluster_names
