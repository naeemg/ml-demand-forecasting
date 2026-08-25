"""
Feature Engineering Module for Demand Forecasting.
Handles time-series lag creation, rolling statistics, and temporal indicators.
"""

import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Extracts temporal features from datetime column."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df


def create_lag_features(df: pd.DataFrame, target_column: str, lags: list) -> pd.DataFrame:
    """Generates historical lag features for time-series modeling."""
    df = df.copy()
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_column: str, windows: list) -> pd.DataFrame:
    """Calculates rolling mean and standard deviation for target metric."""
    df = df.copy()
    for window in windows:
        df[f'{target_column}_roll_mean_{window}'] = df[target_column].shift(1).rolling(window=window).mean()
        df[f'{target_column}_roll_std_{window}'] = df[target_column].shift(1).rolling(window=window).std()
    return df