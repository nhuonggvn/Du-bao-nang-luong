"""Time series forecasting module."""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data into train and test sets.
    
    Args:
        df: Input dataframe with time index
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_index = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    print(f"Train set: {len(train)} records ({train.index.min()} to {train.index.max()})")
    print(f"Test set: {len(test)} records ({test.index.min()} to {test.index.max()})")
    
    return train, test


def baseline_naive(train: pd.Series, test: pd.Series) -> Tuple[np.ndarray, dict]:
    """Naive forecast: use last observation.
    
    Args:
        train: Training series
        test: Test series
        
    Returns:
        Tuple of (predictions, metrics)
    """
    # Last value from train
    last_value = train.iloc[-1]
    predictions = np.full(len(test), last_value)
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': np.mean(np.abs((test - predictions) / test)) * 100
    }
    
    print(f"Naive Baseline - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    return predictions, metrics


def baseline_seasonal_naive(train: pd.Series, test: pd.Series, 
                           seasonal_period: int = 24) -> Tuple[np.ndarray, dict]:
    """Seasonal naive forecast.
    
    Args:
        train: Training series
        test: Test series
        seasonal_period: Seasonal period (e.g., 24 for hourly data)
        
    Returns:
        Tuple of (predictions, metrics)
    """
    predictions = []
    
    for i in range(len(test)):
        # Use value from same time in previous cycle
        if len(train) >= seasonal_period:
            pred = train.iloc[-(seasonal_period - (i % seasonal_period))]
        else:
            pred = train.iloc[-1]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': np.mean(np.abs((test - predictions) / test)) * 100
    }
    
    print(f"Seasonal Naive - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    return predictions, metrics


def forecast_arima(train: pd.Series, test: pd.Series, 
                  order: Tuple[int, int, int] = (1, 0, 1)) -> Tuple[np.ndarray, dict, ARIMA]:
    """ARIMA forecasting.
    
    Args:
        train: Training series
        test: Test series
        order: ARIMA order (p, d, q)
        
    Returns:
        Tuple of (predictions, metrics, model)
    """
    print(f"Fitting ARIMA{order}...")
    
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    
    # Forecast
    predictions = fitted_model.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': np.mean(np.abs((test - predictions) / test)) * 100,
        'AIC': fitted_model.aic,
        'BIC': fitted_model.bic
    }
    
    print(f"ARIMA{order} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, AIC: {fitted_model.aic:.2f}")
    
    return predictions, metrics, fitted_model


def forecast_holt_winters(train: pd.Series, test: pd.Series,
                         seasonal_periods: int = 24) -> Tuple[np.ndarray, dict]:
    """Holt-Winters (Triple Exponential Smoothing) forecasting.
    
    Args:
        train: Training series
        test: Test series
        seasonal_periods: Seasonal period
        
    Returns:
        Tuple of (predictions, metrics)
    """
    print(f"Fitting Holt-Winters with seasonal_periods={seasonal_periods}...")
    
    try:
        model = ExponentialSmoothing(
            train,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        
        predictions = fitted_model.forecast(steps=len(test))
        
        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': np.mean(np.abs((test - predictions) / test)) * 100
        }
        
        print(f"Holt-Winters - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        
        return predictions, metrics
    
    except Exception as e:
        print(f"Holt-Winters failed: {e}")
        # Fallback to simple exponential smoothing
        model = ExponentialSmoothing(train, trend='add', seasonal=None)
        fitted_model = model.fit()
        predictions = fitted_model.forecast(steps=len(test))
        
        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': np.mean(np.abs((test - predictions) / test)) * 100
        }
        
        return predictions, metrics


def compare_models(results: Dict[str, dict]) -> pd.DataFrame:
    """Compare forecasting models.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Comparison dataframe
    """
    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values('MAE')
    
    return comparison
