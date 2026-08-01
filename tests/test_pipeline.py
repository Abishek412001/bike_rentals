import pandas as pd

from src.models.forecaster import forecaster
from src.models.train_pipeline import run_training_pipeline


def test_training_pipeline_execution():
    bundle = run_training_pipeline()

    assert bundle is not None
    assert "model" in bundle
    assert "q10_model" in bundle
    assert "q90_model" in bundle
    assert "metrics" in bundle

    metrics = bundle["metrics"]
    assert metrics["R2"] > 0.5
    assert metrics["RMSE"] > 0.0


def test_forecaster_single_prediction():
    input_df = pd.DataFrame(
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

    res = forecaster.predict_single(input_df)

    assert "predicted_demand" in res
    assert res["predicted_demand"] >= 0
    assert res["q10_demand_bound"] <= res["q90_demand_bound"]
    assert "disclaimer" in res
    assert "Calculated using configurable business assumptions" in res["disclaimer"]
