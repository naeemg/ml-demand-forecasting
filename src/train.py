"""
Model Training Engine for Demand Forecasting.
Trains an XGBoost regression pipeline using synthetic demand data.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features
)


def run_pipeline():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_demand_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Sample data not found. Run data/generate_sample_data.py first.")

    df = pd.read_csv(data_path)
    
    # 2. Apply Feature Engineering
    df = create_time_features(df, date_column="date")
    df = create_lag_features(df, target_column="demand", lags=[1, 7, 14, 30])
    df = create_rolling_features(df, target_column="demand", windows=[7, 30])
    
    # Drop rows with NaN values created by lag/rolling features
    df = df.dropna().reset_index(drop=True)

    # 3. Train/Validation Split (Time-based split)
    feature_cols = [c for c in df.columns if c not in ["date", "item_id", "demand"]]
    X = df[feature_cols]
    y = df["demand"]

    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. Fit Model
    print(f"Training XGBoost Regressor on {len(X_train)} samples...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)

    # 5. Evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mape = mean_absolute_percentage_error(y_val, preds)

    print("\n--- Model Evaluation Results ---")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAPE: {mape:.4f} ({mape * 100:.2f}%)")


if __name__ == "__main__":
    print("Executing ML Demand Forecasting Pipeline...")
    run_pipeline()