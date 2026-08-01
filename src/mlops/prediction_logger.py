import json
import sqlite3
from datetime import UTC, datetime

import pandas as pd

from configs.settings import settings
from src.utils.logger import app_logger


class PredictionLogger:
    """
    Asynchronous prediction logger for auditing inference requests, predictions,
    confidence intervals, model versions, and response latency.
    """

    def __init__(self):
        self.db_path = settings.logs_dir / "predictions_audit.db"
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        request_id TEXT,
                        model_version TEXT,
                        inputs_json TEXT NOT NULL,
                        predicted_demand INTEGER NOT NULL,
                        q10_bound INTEGER,
                        q90_bound INTEGER,
                        estimated_revenue_usd REAL,
                        latency_ms REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            app_logger.warning(f"Prediction audit DB initialization warning: {e}")

    def log_prediction(
        self,
        request_id: str,
        inputs: dict,
        predicted_demand: int,
        q10_bound: int,
        q90_bound: int,
        revenue: float,
        model_version: str,
        latency_ms: float,
    ):
        """
        Logs single inference prediction record to SQLite audit database.
        """
        try:
            ts = datetime.now(UTC).isoformat()
            inputs_str = json.dumps(inputs)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO prediction_audit (
                        timestamp, request_id, model_version, inputs_json,
                        predicted_demand, q10_bound, q90_bound, estimated_revenue_usd, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ts,
                        request_id,
                        model_version,
                        inputs_str,
                        predicted_demand,
                        q10_bound,
                        q90_bound,
                        revenue,
                        latency_ms,
                    ),
                )
                conn.commit()
        except Exception as e:
            app_logger.warning(f"Could not log prediction to audit DB: {e}")

    def get_recent_predictions(self, limit: int = 50) -> pd.DataFrame:
        """
        Retrieves recent prediction records for auditing using parameterized SQL queries.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM prediction_audit ORDER BY id DESC LIMIT ?",
                    conn,
                    params=(int(limit),),
                )
                return df
        except Exception:
            return pd.DataFrame()


prediction_logger = PredictionLogger()
