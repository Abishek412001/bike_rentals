import pandas as pd

from src.features.feature_store import feature_store


def test_feature_store_preparation():
    raw_df = pd.DataFrame(
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

    prepared = feature_store.prepare_features(raw_df, validate=True)

    assert isinstance(prepared, pd.DataFrame)
    assert "hr_sin" in prepared.columns
    assert "hr_cos" in prepared.columns
    assert "commute_interaction" in prepared.columns
