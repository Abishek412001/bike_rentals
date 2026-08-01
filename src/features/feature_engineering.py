import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["season", "yr", "holiday", "workingday", "weathersit", "hum_bins"]
NUM_COLS = [
    "mnth",
    "hr",
    "weekday",
    "temp",
    "hum",
    "windspeed",
    "windspeed_winsorized",
    "is_peak",
    "is_weekend",
    "commute_interaction",
    "temp_sq",
    "feels_like_index",
    "weather_severity",
    "temp_humidity_ratio",
    "weather_temp_interaction",
    "hr_sin",
    "hr_cos",
    "weekday_sin",
    "weekday_cos",
    "mnth_sin",
    "mnth_cos",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain, temporal, weather, and interaction features from raw bike rental dataset.
    """
    df_out = df.copy()

    # Category normalization mappings
    season_map = {1: "springer", 2: "summer", 3: "fall", 4: "winter"}
    weather_map = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}
    yr_map = {0: "2011", 1: "2012"}
    holiday_map = {0: "No", 1: "Yes"}
    work_map = {1: "Working Day", 0: "No work"}

    if df_out["season"].dtype in [np.int64, np.float64, int, float]:
        df_out["season"] = df_out["season"].map(season_map).fillna("springer")
    if df_out["weathersit"].dtype in [np.int64, np.float64, int, float]:
        df_out["weathersit"] = df_out["weathersit"].map(weather_map).fillna("Clear")
    if df_out["yr"].dtype in [np.int64, np.float64, int, float]:
        df_out["yr"] = df_out["yr"].map(yr_map).fillna("2011")
    if df_out["holiday"].dtype in [np.int64, np.float64, int, float]:
        df_out["holiday"] = df_out["holiday"].map(holiday_map).fillna("No")
    if df_out["workingday"].dtype in [np.int64, np.float64, int, float]:
        df_out["workingday"] = df_out["workingday"].map(work_map).fillna("Working Day")

    df_out["mnth"] = df_out["mnth"].astype(float)
    df_out["hr"] = df_out["hr"].astype(int)
    df_out["weekday"] = df_out["weekday"].astype(int)
    df_out["temp"] = df_out["temp"].astype(float)
    df_out["hum"] = df_out["hum"].astype(float)
    df_out["windspeed"] = df_out["windspeed"].astype(float)

    # 1. Temporal cyclic encodings & indicators
    df_out["hr_sin"] = np.sin(2 * np.pi * df_out["hr"] / 24.0)
    df_out["hr_cos"] = np.cos(2 * np.pi * df_out["hr"] / 24.0)
    df_out["weekday_sin"] = np.sin(2 * np.pi * df_out["weekday"] / 7.0)
    df_out["weekday_cos"] = np.cos(2 * np.pi * df_out["weekday"] / 7.0)
    df_out["mnth_sin"] = np.sin(2 * np.pi * df_out["mnth"] / 12.0)
    df_out["mnth_cos"] = np.cos(2 * np.pi * df_out["mnth"] / 12.0)

    df_out["is_peak"] = df_out["hr"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )
    df_out["is_weekend"] = df_out["weekday"].apply(lambda w: 1 if w in [0, 6] else 0)

    # 2. Weather features & winsorization
    p99_wind = df_out["windspeed"].quantile(0.99) if len(df_out) > 10 else 0.6
    df_out["windspeed_winsorized"] = np.clip(df_out["windspeed"], 0.0, p99_wind)
    df_out["temp_sq"] = df_out["temp"] ** 2
    df_out["feels_like_index"] = df_out["temp"] - (0.05 * df_out["windspeed"])

    def get_weather_severity(w):
        severity_dict = {
            "Clear": 1.0,
            "Mist": 2.0,
            "Light Snow": 3.0,
            "Heavy Rain": 4.0,
        }
        return severity_dict.get(str(w), 1.0)

    df_out["weather_severity"] = df_out["weathersit"].apply(get_weather_severity)

    def get_hum_bin(h):
        if h <= 0.3:
            return "Low"
        elif h <= 0.7:
            return "Ideal"
        else:
            return "High"

    df_out["hum_bins"] = df_out["hum"].apply(get_hum_bin)

    # 3. Domain Interactions
    is_working = (df_out["workingday"] == "Working Day").astype(int)
    df_out["commute_interaction"] = df_out["is_peak"] * is_working
    df_out["temp_humidity_ratio"] = df_out["temp"] / (df_out["hum"] + 1e-5)
    df_out["weather_temp_interaction"] = df_out["weather_severity"] * df_out["temp"]

    return df_out


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a robust ColumnTransformer preprocessor pipeline for ML models.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )
