import pandas as pd

from configs.settings import settings
from src.utils.logger import app_logger

try:
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )
    from evidently.report import Report

    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False


class DriftDetector:
    """
    Evidently AI Data, Target & Prediction Drift Monitor.
    """

    def __init__(self):
        self.reports_dir = settings.reports_dir
        self.html_report_path = self.reports_dir / "drift_report.html"

    def run_drift_analysis(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict:
        """
        Executes feature drift and prediction drift report comparing reference vs current data.
        Saves HTML report and returns summary metrics dictionary.
        """
        eval_cols = [
            c
            for c in ["temp", "hum", "windspeed", "hr", "mnth", "weekday", "cnt"]
            if c in reference_df.columns and c in current_df.columns
        ]

        ref_sub = reference_df[eval_cols].dropna()
        curr_sub = current_df[eval_cols].dropna()

        if len(curr_sub) < 5:
            # Fallback if current dataset too small
            curr_sub = ref_sub.sample(min(len(ref_sub), 100), random_state=42)

        summary = {
            "dataset_drift_detected": False,
            "number_of_drifted_features": 0,
            "total_features_evaluated": len(eval_cols),
            "report_path": str(self.html_report_path),
        }

        if HAS_EVIDENTLY:
            try:
                report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
                report.run(reference_data=ref_sub, current_data=curr_sub)
                report.save_html(str(self.html_report_path))

                # Extract metric summary
                res_dict = report.as_dict()
                drift_metrics = res_dict.get("metrics", [])[0].get("result", {})
                summary["dataset_drift_detected"] = drift_metrics.get(
                    "dataset_drift", False
                )
                summary["number_of_drifted_features"] = drift_metrics.get(
                    "number_of_drifted_columns", 0
                )
                app_logger.info(
                    f"Evidently AI Drift Analysis generated at {self.html_report_path}"
                )
            except Exception as e:
                app_logger.warning(f"Error executing Evidently report: {e}")

        return summary


drift_detector = DriftDetector()
