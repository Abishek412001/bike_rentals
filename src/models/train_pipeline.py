import urllib.request

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from configs.settings import settings
from src.features.feature_engineering import CAT_COLS, NUM_COLS, build_preprocessor
from src.features.feature_store import feature_store
from src.mlops.mlflow_tracker import mlflow_tracker
from src.mlops.registry import model_registry
from src.models.evaluator import (
    calculate_metrics,
    generate_all_reports_figures,
    generate_model_card,
)
from src.utils.logger import app_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_bike_data() -> pd.DataFrame:
    """
    Retrieves dataset from UCI repository or falls back to realistic synthetic generator.
    """
    url = "https://raw.githubusercontent.com/byuidaho-cs/datasets/main/bike-sharing/hour.csv"
    csv_file = settings.base_dir / "hour.csv"

    if not csv_file.exists():
        app_logger.info("Downloading UCI Bike Sharing dataset...")
        try:
            urllib.request.urlretrieve(url, csv_file)  # nosec B310
            app_logger.info("Download successful.")
        except Exception as e:
            app_logger.warning(
                f"Could not download dataset: {e}. Generating realistic synthetic dataset..."
            )
            return generate_synthetic_data()

    try:
        df = pd.read_csv(csv_file)
        return feature_store.prepare_features(df, validate=True)
    except Exception as e:
        app_logger.warning(f"Error loading CSV: {e}. Generating synthetic dataset...")
        return generate_synthetic_data()


def generate_synthetic_data(num_samples: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    seasons = np.random.choice(
        ["springer", "summer", "fall", "winter"], size=num_samples
    )
    yrs = np.random.choice(["2011", "2012"], size=num_samples)
    mnths = np.random.randint(1, 13, size=num_samples).astype(float)
    hrs = np.random.randint(0, 24, size=num_samples)
    holidays = np.random.choice(["No", "Yes"], p=[0.97, 0.03], size=num_samples)
    weekdays = np.random.randint(0, 7, size=num_samples)
    workingdays = np.random.choice(
        ["Working Day", "No work"], p=[0.7, 0.3], size=num_samples
    )
    weathers = np.random.choice(
        ["Clear", "Mist", "Light Snow", "Heavy Rain"],
        p=[0.6, 0.25, 0.1, 0.05],
        size=num_samples,
    )
    temps = np.random.uniform(0.1, 0.9, size=num_samples)
    hums = np.random.uniform(0.1, 0.9, size=num_samples)
    winds = np.random.uniform(0.0, 0.6, size=num_samples)

    is_peak = np.array([1 if (7 <= h <= 9) or (17 <= h <= 19) else 0 for h in hrs])
    is_work = np.array([1 if w == "Working Day" else 0 for w in workingdays])
    commute_interaction = is_peak * is_work

    base_cnt = (
        40
        + hrs * 9
        + temps * 240
        - hums * 90
        + commute_interaction * 310
        + (yrs == "2012") * 45
    )
    cnt = np.clip(base_cnt + np.random.normal(0, 28, size=num_samples), 1, None)

    raw_df = pd.DataFrame(
        {
            "season": seasons,
            "yr": yrs,
            "mnth": mnths,
            "hr": hrs,
            "holiday": holidays,
            "weekday": weekdays,
            "workingday": workingdays,
            "weathersit": weathers,
            "temp": temps,
            "hum": hums,
            "windspeed": winds,
            "cnt": cnt,
        }
    )

    return feature_store.prepare_features(raw_df, validate=True)


def optimize_xgboost_optuna(X_prep, y_log, tscv) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }

        rmse_scores = []
        for train_idx, val_idx in tscv.split(X_prep):
            X_tr, X_va = X_prep[train_idx], X_prep[val_idx]
            y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = np.expm1(model.predict(X_va))
            actuals = np.expm1(y_va)
            rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
            rmse_scores.append(rmse)

        mean_rmse = float(np.mean(rmse_scores))
        mlflow_tracker.log_trial("xgboost_optuna_trial", params, {"cv_rmse": mean_rmse})
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8, timeout=60)
    app_logger.info(f"Optuna Best XGBoost Hyperparameters: {study.best_params}")
    return study.best_params


def run_training_pipeline() -> dict:
    app_logger.info("Starting Enterprise Model Training Pipeline...")
    df = get_bike_data()

    X = df[CAT_COLS + NUM_COLS]
    y_log = np.log1p(df["cnt"])

    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)
    all_feature_names = feature_store.get_feature_names(preprocessor)

    tscv = TimeSeriesSplit(n_splits=5)

    candidate_models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            n_estimators=150, depth=6, learning_rate=0.08, random_seed=42, verbose=0
        ),
        "XGBoost": XGBRegressor(
            n_estimators=180, max_depth=6, learning_rate=0.08, random_state=42
        ),
    }

    model_benchmark_results = []
    trained_models = {}

    for model_name, candidate in candidate_models.items():
        rmse_list, mae_list, mape_list, r2_list = [], [], [], []

        for train_idx, val_idx in tscv.split(X_prep):
            X_tr, X_va = X_prep[train_idx], X_prep[val_idx]
            y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]

            candidate.fit(X_tr, y_tr)
            preds_log = candidate.predict(X_va)
            actuals = np.expm1(y_va)
            preds = np.clip(np.expm1(preds_log), 0, None)

            m = calculate_metrics(actuals.values, preds)
            rmse_list.append(m["rmse"])
            mae_list.append(m["mae"])
            mape_list.append(m["mape"])
            r2_list.append(m["r2"])

        avg_metrics = {
            "R2": float(np.mean(r2_list)),
            "RMSE": float(np.mean(rmse_list)),
            "MAE": float(np.mean(mae_list)),
            "MAPE": float(np.mean(mape_list)),
        }

        app_logger.info(
            f" -> {model_name} Benchmark: R2={avg_metrics['R2']:.4f}, RMSE={avg_metrics['RMSE']:.2f}"
        )
        mlflow_tracker.log_trial(model_name, {}, avg_metrics)

        model_benchmark_results.append({"Model": model_name, **avg_metrics})

        candidate.fit(X_prep, y_log)
        trained_models[model_name] = candidate

    benchmark_df = pd.DataFrame(model_benchmark_results).sort_values(by="RMSE")
    champion_name = benchmark_df.iloc[0]["Model"]
    challenger_name = benchmark_df.iloc[1]["Model"]

    app_logger.info(
        f"Selected Champion Model: {champion_name} | Challenger Model: {challenger_name}"
    )

    champion_model = trained_models[champion_name]
    if champion_name == "XGBoost":
        best_params = optimize_xgboost_optuna(X_prep, y_log, tscv)
        champion_model = XGBRegressor(**best_params)
        champion_model.fit(X_prep, y_log)

    challenger_model = trained_models[challenger_name]

    # Quantile Regressors
    q10_model = XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        random_state=42,
    )
    q90_model = XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        random_state=42,
    )
    q10_model.fit(X_prep, y_log)
    q90_model.fit(X_prep, y_log)

    champ_metrics = (
        benchmark_df[benchmark_df["Model"] == champion_name].iloc[0].to_dict()
    )
    chall_metrics = (
        benchmark_df[benchmark_df["Model"] == challenger_name].iloc[0].to_dict()
    )

    champion_bundle = {
        "model": champion_model,
        "q10_model": q10_model,
        "q90_model": q90_model,
        "pipeline": preprocessor,
        "metrics": champ_metrics,
        "model_name": champion_name,
        "feature_names": all_feature_names,
    }

    challenger_bundle = {
        "model": challenger_model,
        "pipeline": preprocessor,
        "metrics": chall_metrics,
        "model_name": challenger_name,
        "feature_names": all_feature_names,
    }

    model_registry.save_champion(
        champion_bundle, champ_metrics, champion_name, all_feature_names
    )
    model_registry.save_challenger(challenger_bundle, chall_metrics, challenger_name)

    # Save root bundle for backward compatibility
    root_bundle_path = settings.base_dir / "bike_demand_model_bundle.pkl"
    joblib.dump(champion_bundle, root_bundle_path)

    # Auto-generate Model Card & Reports
    generate_model_card(champion_name, champ_metrics, all_feature_names, len(df))
    generate_all_reports_figures(
        df=df,
        X_prep=X_prep,
        model=champion_model,
        preprocessor=preprocessor,
        feature_names=all_feature_names,
        model_comparison_df=benchmark_df,
    )

    return champion_bundle
