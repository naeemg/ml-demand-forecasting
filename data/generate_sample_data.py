"""
Synthetic Data Generator for Demand Forecasting.
Generates realistic multi-item daily sales data with seasonality, trend, and noise.
"""

import os
import numpy as np
import pandas as pd


def generate_demand_dataset(
    num_items: int = 5,
    days: int = 730,
    start_date: str = "2024-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """Generates synthetic time-series transaction data for multi-item inventory."""
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=days, freq="D")
    records = []

    for item_id in range(1, num_items + 1):
        base_demand = np.random.randint(50, 200)
        trend = np.linspace(0, 30, days)
        
        # Seasonal component (weekly + annual cycles)
        day_of_year = date_range.dayofyear
        day_of_week = date_range.dayofweek
        seasonality = 15 * np.sin(2 * np.pi * day_of_year / 365.25) + 10 * (day_of_week >= 5)
        
        # Random noise
        noise = np.random.normal(0, 8, days)
        
        # Calculate final daily demand
        demand = base_demand + trend + seasonality + noise
        demand = np.maximum(0, demand).astype(int)  # Demand cannot be negative

        for date, qty in zip(date_range, demand):
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "item_id": f"ITEM_{item_id:03d}",
                "demand": qty
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Define output path
    output_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(output_dir, "sample_demand_data.csv")

    print("Generating synthetic demand forecasting dataset...")
    df = generate_demand_dataset()
    df.to_csv(file_path, index=False)
    print(f"Successfully saved {len(df)} rows to: {file_path}")