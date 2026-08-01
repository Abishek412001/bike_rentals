import pandas as pd

from src.monitoring.drift_detector import drift_detector


def test_drift_analysis_execution():
    ref_df = pd.DataFrame(
        {
            "temp": [0.5, 0.6, 0.7, 0.55],
            "hum": [0.4, 0.5, 0.45, 0.5],
            "windspeed": [0.1, 0.2, 0.15, 0.1],
            "hr": [8, 12, 18, 20],
            "cnt": [120, 300, 450, 200],
        }
    )

    curr_df = pd.DataFrame(
        {
            "temp": [0.85, 0.9, 0.88, 0.92],
            "hum": [0.8, 0.85, 0.9, 0.82],
            "windspeed": [0.4, 0.5, 0.45, 0.5],
            "hr": [8, 12, 18, 20],
            "cnt": [50, 80, 60, 40],
        }
    )

    res = drift_detector.run_drift_analysis(ref_df, curr_df)
    assert isinstance(res, dict)
    assert "dataset_drift_detected" in res
    assert "report_path" in res
