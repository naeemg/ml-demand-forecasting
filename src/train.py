"""
Model Training Engine for Demand Forecasting.
Trains an XGBoost regression pipeline and outputs evaluation metrics.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from feature_engineering import create_time_features, create_lag_features, create_rolling_features


def train_demand_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    """Initializes and fits XGBoost Regressor for demand prediction."""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mape = mean_absolute_percentage_error(y_val, preds)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAPE: {mape:.4f}")
    
    return model


if __name__ == "__main__":
    print("Executing ML Demand Forecasting Training Pipeline...")