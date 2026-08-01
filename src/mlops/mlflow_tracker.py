import os

from configs.settings import settings
from src.utils.logger import app_logger

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTracker:
    """
    Local file-based MLflow experiment tracker.
    Logs hyperparameters, cross-validation metrics, Optuna trials, and artifacts.
    """

    def __init__(self, experiment_name: str = "bike_rental_demand_forecasting"):
        self.experiment_name = experiment_name
        self.mlruns_dir = settings.base_dir / "mlruns"
        self.mlruns_dir.mkdir(exist_ok=True)

        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(f"file:///{self.mlruns_dir.as_posix()}")
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                app_logger.warning(f"Could not initialize MLflow experiment: {e}")

    def log_trial(
        self, model_name: str, params: dict, metrics: dict, artifacts: dict = None
    ):
        """
        Logs a single training trial / model evaluation run.
        """
        if not HAS_MLFLOW:
            return

        try:
            with mlflow.start_run(run_name=f"train_{model_name}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                if artifacts:
                    for key, path_str in artifacts.items():
                        if os.path.exists(path_str):
                            mlflow.log_artifact(path_str)
        except Exception as e:
            app_logger.warning(f"MLflow log_trial error: {e}")


mlflow_tracker = MLflowTracker()
