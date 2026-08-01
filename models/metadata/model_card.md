# Model Card: CatBoost Bike Rental Demand Predictor

## Model Details
- **Architecture:** CatBoost Regressor with XGBoost Quantile Regression Bounds
- **Model Version:** 3.0.0
- **Date Trained:** 2026-08-01 08:36:26 UTC
- **Framework:** Scikit-Learn / CatBoost / XGBoost / LightGBM
- **Target Variable:** Hourly Bike Rental Count (`cnt`) with log1p transformation

## Intended Use & Business Purpose
- **Primary Use Case:** Hourly micro-mobility demand forecasting, station fleet rebalancing, and stockout risk management.
- **Target Users:** Logistics dispatch managers, smart city transit planners, and fleet rebalancing operators.
- **Operational Assumption Disclaimer:** All financial predictions, staffing requirements, and rebalancing truck allocations are calculated using configurable business assumptions from `configs/business.yaml`.

## Performance Benchmarks (5-Fold TimeSeriesSplit Cross Validation)
- **Mean R² Score:** 0.9596
- **Mean RMSE:** 30.64 bikes
- **Mean MAE:** 24.20 bikes
- **Mean MAPE:** 20.98%
- **Training Samples:** 5,000 hourly records

## Input Features (38 features)
- **Temporal Features:** `hr_sin`, `hr_cos`, `weekday_sin`, `weekday_cos`, `mnth_sin`, `mnth_cos`, `is_peak`, `is_weekend`
- **Weather & Comfort Features:** `temp`, `hum`, `windspeed`, `temp_sq`, `feels_like_index`, `weather_severity`, `hum_bins`
- **Interaction Features:** `commute_interaction`, `temp_humidity_ratio`, `weather_temp_interaction`

## Operational Considerations & Limitations
- Peak demand bounds ($P_10$ and $P_90$) rely on quantile loss functions (`reg:quantileerror`).
- Predictions are bounded for normalized temperatures ($0.0$ to $1.0$) and relative humidity ($0.0$ to $1.0$).
- Extreme weather anomalies (e.g. severe blizzard or hurricane evacuations) fall outside historical distribution bounds.
