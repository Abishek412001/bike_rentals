import json
import shutil
import subprocess  # nosec B404
from datetime import UTC, datetime

import joblib

from configs.settings import settings
from src.utils.logger import app_logger


def get_git_commit_hash() -> str:
    try:
        commit = subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "local_dev"


class ModelRegistry:
    """
    Enterprise Champion-Challenger Model Registry manager with rollback support.
    """

    def __init__(self):
        self.champion_dir = settings.champion_dir
        self.challenger_dir = settings.challenger_dir
        self.metadata_path = settings.metadata_dir / "metadata.json"

    def save_champion(
        self, bundle: dict, metrics: dict, model_name: str, feature_names: list
    ):
        bundle_path = self.champion_dir / "model_bundle.pkl"
        joblib.dump(bundle, bundle_path)
        app_logger.info(f"Saved Champion model bundle ({model_name}) to {bundle_path}")

        metadata = self.load_metadata()
        metadata.update(
            {
                "version": settings.app_version,
                "champion_model": model_name,
                "champion_metrics": metrics,
                "training_timestamp": datetime.now(UTC).isoformat(),
                "git_commit": get_git_commit_hash(),
                "feature_names": feature_names,
                "pipeline_version": settings.app_version,
            }
        )
        self._write_metadata(metadata)

    def save_challenger(self, bundle: dict, metrics: dict, model_name: str):
        bundle_path = self.challenger_dir / "model_bundle.pkl"
        joblib.dump(bundle, bundle_path)
        app_logger.info(
            f"Saved Challenger model bundle ({model_name}) to {bundle_path}"
        )

        metadata = self.load_metadata()
        metadata.update({"challenger_model": model_name, "challenger_metrics": metrics})
        self._write_metadata(metadata)

    def rollback_to_challenger(self) -> dict:
        """
        Executes instant rollback: swaps Challenger model bundle into Champion position.
        """
        champ_bundle_path = self.champion_dir / "model_bundle.pkl"
        chall_bundle_path = self.challenger_dir / "model_bundle.pkl"

        if not chall_bundle_path.exists():
            raise FileNotFoundError(
                "No challenger model bundle available for rollback."
            )

        shutil.copy(chall_bundle_path, champ_bundle_path)
        app_logger.info("Successfully rolled back Champion model to Challenger bundle.")

        metadata = self.load_metadata()
        old_champ = metadata.get("champion_model")
        old_chall = metadata.get("challenger_model")

        metadata["champion_model"] = old_chall
        metadata["challenger_model"] = old_champ
        metadata["last_rollback_timestamp"] = datetime.now(UTC).isoformat()

        self._write_metadata(metadata)
        return metadata

    def load_champion(self) -> dict:
        bundle_path = self.champion_dir / "model_bundle.pkl"
        if not bundle_path.exists():
            root_bundle = settings.base_dir / "bike_demand_model_bundle.pkl"
            if root_bundle.exists():
                return joblib.load(root_bundle)
            raise FileNotFoundError(f"Champion model bundle not found at {bundle_path}")
        return joblib.load(bundle_path)

    def load_challenger(self) -> dict:
        bundle_path = self.challenger_dir / "model_bundle.pkl"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Challenger model bundle not found at {bundle_path}"
            )
        return joblib.load(bundle_path)

    def load_metadata(self) -> dict:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                app_logger.warning(f"Error reading metadata.json: {e}")
        return {
            "version": settings.app_version,
            "champion_model": "CatBoost",
            "champion_metrics": {},
            "challenger_model": "XGBoost",
            "challenger_metrics": {},
            "training_timestamp": "N/A",
            "git_commit": get_git_commit_hash(),
            "pipeline_version": settings.app_version,
        }

    def _write_metadata(self, metadata: dict):
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


model_registry = ModelRegistry()
