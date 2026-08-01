import pandas as pd
import pandera as pa
import pytest

from src.utils.validation import validate_input_data


def test_valid_input_data():
    valid_df = pd.DataFrame(
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

    validated = validate_input_data(valid_df)
    assert len(validated) == 1


def test_invalid_temperature_bounds():
    invalid_df = pd.DataFrame(
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
                "temp": 1.85,  # Out of range 0-1
                "hum": 0.45,
                "windspeed": 0.15,
            }
        ]
    )

    with pytest.raises(pa.errors.SchemaError):
        validate_input_data(invalid_df)
