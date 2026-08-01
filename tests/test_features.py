import numpy as np
import pandas as pd

from src.features.feature_engineering import build_preprocessor, engineer_features


def test_engineer_features_columns():
    df = pd.DataFrame(
        [
            {
                "season": 2,
                "yr": 1,
                "mnth": 6,
                "hr": 18,
                "holiday": 0,
                "weekday": 2,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.65,
                "hum": 0.45,
                "windspeed": 0.15,
            }
        ]
    )

    feat_df = engineer_features(df)

    # Check cyclic encodings present
    assert "hr_sin" in feat_df.columns
    assert "hr_cos" in feat_df.columns
    assert "weekday_sin" in feat_df.columns
    assert "mnth_sin" in feat_df.columns

    # Check bounds
    assert -1.0 <= feat_df["hr_sin"].iloc[0] <= 1.0
    assert -1.0 <= feat_df["hr_cos"].iloc[0] <= 1.0

    # Check interaction features
    assert feat_df["is_peak"].iloc[0] == 1
    assert feat_df["commute_interaction"].iloc[0] == 1
    assert feat_df["hum_bins"].iloc[0] == "Ideal"


def test_preprocessor_transformation():
    df = pd.DataFrame(
        [
            {
                "season": "summer",
                "yr": "2012",
                "mnth": 6.0,
                "hr": 18,
                "holiday": "No",
                "weekday": 2,
                "workingday": "Working Day",
                "weathersit": "Clear",
                "temp": 0.65,
                "hum": 0.45,
                "windspeed": 0.15,
            }
        ]
    )

    feat_df = engineer_features(df)
    prep = build_preprocessor()
    prep_data = prep.fit_transform(feat_df)

    assert isinstance(prep_data, np.ndarray)
    assert prep_data.shape[0] == 1
    assert prep_data.shape[1] > 10
