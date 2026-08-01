import pandas as pd

from src.features.feature_engineering import CAT_COLS, NUM_COLS, engineer_features
from src.utils.validation import validate_input_data


class FeatureStore:
    """
    Centralized Feature Store for offline training and online inference feature parity.
    Ensures identical feature transformations and schema validation across pipelines.
    """

    def __init__(self):
        self.cat_cols = CAT_COLS
        self.num_cols = NUM_COLS

    def prepare_features(
        self, raw_df: pd.DataFrame, validate: bool = True
    ) -> pd.DataFrame:
        """
        Validates raw input data and engineers temporal, weather, and interaction features.
        """
        if validate:
            validated_df = validate_input_data(raw_df)
        else:
            validated_df = raw_df.copy()

        engineered_df = engineer_features(validated_df)
        return engineered_df

    def get_feature_names(self, preprocessor) -> list:
        """
        Returns full list of one-hot encoded and numerical feature names.
        """
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(self.cat_cols).tolist()
        return cat_names + self.num_cols


feature_store = FeatureStore()
