from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class PlatformSettings(BaseSettings):
    app_name: str = "Enterprise Bike Rental Demand Platform"
    app_version: str = "3.0.0"
    environment: str = "production"
    debug: bool = False

    # Base Paths
    base_dir: Path = BASE_DIR
    models_dir: Path = BASE_DIR / "models"
    champion_dir: Path = BASE_DIR / "models" / "champion"
    challenger_dir: Path = BASE_DIR / "models" / "challenger"
    metadata_dir: Path = BASE_DIR / "models" / "metadata"
    reports_dir: Path = BASE_DIR / "reports"
    figures_dir: Path = BASE_DIR / "reports" / "figures"
    configs_dir: Path = BASE_DIR / "configs"
    logs_dir: Path = BASE_DIR / "logs"

    # Config File Paths
    business_config_path: Path = BASE_DIR / "configs" / "business.yaml"
    model_config_path: Path = BASE_DIR / "configs" / "model.yaml"
    training_config_path: Path = BASE_DIR / "configs" / "training.yaml"
    api_config_path: Path = BASE_DIR / "configs" / "api.yaml"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def _read_yaml(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_business_config(self) -> dict:
        return self._read_yaml(self.business_config_path) or {
            "financial": {"average_rental_price_usd": 2.75},
            "logistics": {
                "truck_capacity_bikes": 25,
                "staff_capacity_bikes": 50,
                "total_fleet_capacity": 500,
            },
            "disclaimer": "Calculated using configurable business assumptions.",
        }

    def load_model_config(self) -> dict:
        return self._read_yaml(self.model_config_path)

    def load_training_config(self) -> dict:
        return self._read_yaml(self.training_config_path)

    def load_api_config(self) -> dict:
        return self._read_yaml(self.api_config_path)


settings = PlatformSettings()

for path in [
    settings.models_dir,
    settings.champion_dir,
    settings.challenger_dir,
    settings.metadata_dir,
    settings.reports_dir,
    settings.figures_dir,
    settings.configs_dir,
    settings.logs_dir,
]:
    path.mkdir(parents=True, exist_ok=True)
