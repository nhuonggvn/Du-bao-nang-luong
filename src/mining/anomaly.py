"""Anomaly detection module for household power consumption."""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

def detect_anomalies_isolation_forest(df: pd.DataFrame, features: list, contamination: float = 0.05) -> pd.DataFrame:
    """Detect anomalies using Isolation Forest.
    
    Args:
        df: DataFrame containing daily profiles
        features: List of feature column names to use
        contamination: Expected proportion of outliers
        
    Returns:
        DataFrame with added 'is_anomaly' and 'anomaly_score' columns
    """
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Initialize model
    iso_forest = IsolationForest(
        contamination=contamination, 
        random_state=42,
        n_estimators=100
    )
    
    # Fit and predict (-1 for anomalies, 1 for normal)
    X = result_df[features].fillna(0)
    preds = iso_forest.fit_predict(X)
    scores = iso_forest.score_samples(X)
    
    # Convert predictions to boolean (True for anomaly)
    result_df['is_anomaly'] = preds == -1
    result_df['anomaly_score'] = scores
    
    return result_df

def plot_anomalies_profile(df: pd.DataFrame, date_col: str = None) -> plt.Figure:
    """Plot daily total consumption highlighting anomalies."""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    normal_data = df[~df['is_anomaly']]
    anomaly_data = df[df['is_anomaly']]
    
    if date_col and date_col in df.columns:
        x_norm = normal_data[date_col]
        x_anom = anomaly_data[date_col]
    else:
        x_norm = normal_data.index
        x_anom = anomaly_data.index
        
    ax.scatter(x_norm, normal_data['total_consumption'], color='blue', alpha=0.5, label='Bình thường')
    ax.scatter(x_anom, anomaly_data['total_consumption'], color='red', s=100, label='Bất thường')
    
    ax.set_title('Phát hiện ngày tiêu thụ bất thường (Anomaly Detection)', fontsize=14)
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Tổng điện năng tiêu thụ (hàng ngày)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def get_anomaly_insights(df: pd.DataFrame) -> str:
    """Generate insights about detected anomalies."""
    n_total = len(df)
    n_anomalies = df['is_anomaly'].sum()
    pct_anomalies = (n_anomalies / n_total) * 100
    
    anomalies = df[df['is_anomaly']]
    avg_normal = df[~df['is_anomaly']]['total_consumption'].mean()
    avg_anom = anomalies['total_consumption'].mean()
    
    insight = f"Phân tích Bất thường:\\n"
    insight += f"- Tổng số ngày quan sát: {n_total}\\n"
    insight += f"- Số ngày bất thường phát hiện được: {n_anomalies} ({pct_anomalies:.2f}%)\\n"
    insight += f"- Mức tiêu thụ trung bình ngày bình thường: {avg_normal:.2f}\\n"
    insight += f"- Mức tiêu thụ trung bình ngày bất thường: {avg_anom:.2f}\\n"
    
    if avg_anom > avg_normal:
        insight += "=> Đa số các điểm bất thường là những ngày tiêu thụ CAO đột biến (có thể do lễ tết, thời tiết cực đoan, hoặc hệ thống làm mát/sưởi ấm hoạt động quá công suất)."
    else:
        insight += "=> Đa số các điểm bất thường là những ngày tiêu thụ THẤP đột biến (có thể do hộ gia đình đi vắng, mất điện hoặc thiết bị hỏng)."
        
    return insight
