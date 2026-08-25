# Machine Learning Demand Forecasting Framework (`ml-demand-forecasting`)

## Executive Summary
This repository provides an enterprise-grade demand forecasting framework designed to optimize supply chain logistics and inventory allocation. By applying advanced supervised machine learning algorithms (XGBoost, Random Forest) and time-series decomposition to historical transaction data, the system reduces stockouts and holding costs across multi-echelon retail networks.

## Architecture & Pipeline
1. **Data Ingestion & Invalidation Handling:** Automated pipeline for missing value imputation and outlier removal.
2. **Time-Series Feature Engineering:** Generates lag variables, rolling window statistics, and seasonal calendar indicators.
3. **Model Engine:** Trains ensemble regression models optimized against Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).
4. **Export & Forecasting:** Outputs rolling 30-to-90 day predictive demand curves with confidence bounds.

## Quick Start
```bash
# Clone the repository
git clone [https://github.com/naeemg/ml-demand-forecasting.git](https://github.com/naeemg/ml-demand-forecasting.git)
cd ml-demand-forecasting

# Install dependencies
pip install -r requirements.txt
i c re
# Run model training
python src/train.py