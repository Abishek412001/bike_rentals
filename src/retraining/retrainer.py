from datetime import UTC, datetime

from configs.settings import settings
from src.mlops.registry import model_registry
from src.models.train_pipeline import run_training_pipeline
from src.utils.logger import app_logger


class AutomatedRetrainer:
    """
    Automated Retraining, Champion-Challenger evaluation, and Rollback pipeline.
    """

    def __init__(self):
        self.training_config = settings.load_training_config()
        self.thresholds = self.training_config.get("thresholds", {})
        self.min_improvement = self.thresholds.get(
            "min_r2_improvement_for_promotion", 0.005
        )

    def execute_retraining(self) -> dict:
        """
        Executes retraining, evaluates performance against active Champion,
        and automatically promotes if performance improves.
        """
        app_logger.info("Executing automated model retraining pipeline...")

        old_metadata = model_registry.load_metadata()
        old_rmse = old_metadata.get("champion_metrics", {}).get("RMSE", 999.0)

        # Run fresh training
        new_bundle = run_training_pipeline()
        new_rmse = new_bundle["metrics"].get("RMSE", 999.0)

        is_promoted = False
        if new_rmse < (old_rmse - self.min_improvement):
            is_promoted = True
            app_logger.info(
                f"Retraining Promoted: New RMSE ({new_rmse:.2f}) beats previous Champion RMSE ({old_rmse:.2f})."
            )
        else:
            app_logger.info(
                f"Retraining Retained Current Champion: New RMSE ({new_rmse:.2f}) vs Previous ({old_rmse:.2f})."
            )

        return {
            "retraining_timestamp": datetime.now(UTC).isoformat(),
            "previous_rmse": old_rmse,
            "new_rmse": new_rmse,
            "promoted_to_champion": is_promoted,
            "champion_model": new_bundle.get("model_name", "CatBoost"),
        }

    def rollback(self) -> dict:
        """
        Triggers instant rollback to previous Challenger model.
        """
        return model_registry.rollback_to_challenger()


retrainer = AutomatedRetrainer()
