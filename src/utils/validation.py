import pandas as pd
import pandera as pa

BikeDemandInputSchema = pa.DataFrameSchema(
    columns={
        "season": pa.Column(object, nullable=False),
        "yr": pa.Column(object, nullable=False),
        "mnth": pa.Column(float, checks=pa.Check.in_range(1.0, 12.0), nullable=False),
        "hr": pa.Column(int, checks=pa.Check.in_range(0, 23), nullable=False),
        "holiday": pa.Column(object, nullable=False),
        "weekday": pa.Column(int, checks=pa.Check.in_range(0, 6), nullable=False),
        "workingday": pa.Column(object, nullable=False),
        "weathersit": pa.Column(object, nullable=False),
        "temp": pa.Column(float, checks=pa.Check.in_range(0.0, 1.0), nullable=False),
        "hum": pa.Column(float, checks=pa.Check.in_range(0.0, 1.0), nullable=False),
        "windspeed": pa.Column(
            float, checks=pa.Check.in_range(0.0, 1.0), nullable=False
        ),
    },
    coerce=True,
    strict=False,
)


def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates input DataFrame against Pandera DataFrameSchema.
    Raises pa.errors.SchemaError if validation fails.
    """
    df_copy = df.copy()
    if "mnth" in df_copy.columns:
        df_copy["mnth"] = df_copy["mnth"].astype(float)
    if "hr" in df_copy.columns:
        df_copy["hr"] = df_copy["hr"].astype(int)
    if "weekday" in df_copy.columns:
        df_copy["weekday"] = df_copy["weekday"].astype(int)
    if "temp" in df_copy.columns:
        df_copy["temp"] = df_copy["temp"].astype(float)
    if "hum" in df_copy.columns:
        df_copy["hum"] = df_copy["hum"].astype(float)
    if "windspeed" in df_copy.columns:
        df_copy["windspeed"] = df_copy["windspeed"].astype(float)

    return BikeDemandInputSchema.validate(df_copy)
