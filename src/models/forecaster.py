import numpy as np
import pandas as pd

from configs.settings import settings
from src.features.feature_engineering import engineer_features
from src.mlops.registry import model_registry
from src.utils.logger import app_logger


class DemandForecaster:
    """
    Multi-horizon demand forecasting and operational recommendation engine.
    Derives revenue, fleet utilization, and rebalancing recommendations
    using configurable business assumptions from configs/business.yaml.
    """

    def __init__(self, champion_bundle: dict = None):
        if champion_bundle is not None:
            self.bundle = champion_bundle
        else:
            try:
                self.bundle = model_registry.load_champion()
            except Exception as e:
                app_logger.warning(
                    f"Could not load champion model: {e}. Will load when available."
                )
                self.bundle = None

        self.business_config = settings.load_business_config()

    def set_bundle(self, bundle: dict):
        self.bundle = bundle

    def predict_single(self, input_df: pd.DataFrame) -> dict:
        """
        Executes single scenario demand prediction with 90% confidence intervals
        and calculates operational recommendations using configurable business assumptions.
        """
        if self.bundle is None:
            self.bundle = model_registry.load_champion()

        pipeline = self.bundle["pipeline"]
        model = self.bundle["model"]
        q10_model = self.bundle.get("q10_model", model)
        q90_model = self.bundle.get("q90_model", model)

        feat_df = engineer_features(input_df)
        prep_data = pipeline.transform(feat_df)

        pred_log = model.predict(prep_data)[0]
        q10_log = q10_model.predict(prep_data)[0]
        q90_log = q90_model.predict(prep_data)[0]

        pred_demand = int(np.clip(np.expm1(pred_log), 0, None))
        q10_demand = int(np.clip(np.expm1(q10_log), 0, None))
        q90_demand = max(q10_demand, int(np.clip(np.expm1(q90_log), 0, None)))

        # Configurable business assumptions calculations
        financial_cfg = self.business_config.get("financial", {})
        logistics_cfg = self.business_config.get("logistics", {})
        thresholds_cfg = self.business_config.get("thresholds", {})

        price = financial_cfg.get("average_rental_price_usd", 2.75)
        fleet_cap = logistics_cfg.get("total_fleet_capacity", 500)
        truck_cap = logistics_cfg.get("truck_capacity_bikes", 25)
        staff_cap = logistics_cfg.get("staff_capacity_bikes", 50)

        estimated_revenue = round(pred_demand * price, 2)
        fleet_utilization_pct = round(min(100.0, (pred_demand / fleet_cap) * 100.0), 1)

        # Operational rebalancing recommendations
        excess_or_shortage = max(0, pred_demand - int(fleet_cap * 0.8))
        trucks_needed = (
            int(np.ceil(excess_or_shortage / truck_cap))
            if excess_or_shortage > 0
            else 0
        )
        staff_needed = int(np.ceil(pred_demand / staff_cap))

        high_thresh = thresholds_cfg.get("high_demand_threshold", 400)
        mod_thresh = thresholds_cfg.get("moderate_demand_threshold", 200)

        if pred_demand > high_thresh:
            stockout_risk = "High"
        elif pred_demand > mod_thresh:
            stockout_risk = "Moderate"
        else:
            stockout_risk = "Low"

        is_peak = bool(feat_df["is_peak"].iloc[0])
        is_work = (
            (input_df["workingday"].iloc[0] == "Working Day")
            if "workingday" in input_df.columns
            else False
        )

        recommendations = []
        if is_peak and is_work:
            recommendations.append(
                "⚠️ Commute Peak Active: Pre-position rebalancing trucks at key transit stations 90 mins prior."
            )
        if trucks_needed > 0:
            recommendations.append(
                f"🚚 Dispatch {trucks_needed} rebalancing truck(s) to relocate excess bike stock."
            )
        if stockout_risk == "High":
            recommendations.append(
                "🚨 Stockout Risk High: Enable dynamic surge fulfillment pricing / rapid dock clearing."
            )

        if not recommendations:
            recommendations.append(
                "✅ Operations Normal: Standard distribution levels adequate."
            )

        return {
            "predicted_demand": pred_demand,
            "q10_demand_bound": q10_demand,
            "q90_demand_bound": q90_demand,
            "confidence_interval_pct": 90,
            "estimated_revenue_usd": estimated_revenue,
            "fleet_utilization_pct": fleet_utilization_pct,
            "trucks_recommended": trucks_needed,
            "staff_recommended": staff_needed,
            "stockout_risk_level": stockout_risk,
            "operational_recommendations": recommendations,
            "disclaimer": self.business_config.get(
                "disclaimer", "Calculated using configurable business assumptions."
            ),
        }

    def predict_multi_horizon(
        self, base_input: pd.DataFrame, hours_ahead: int = 24
    ) -> pd.DataFrame:
        """
        Generates multi-horizon forecasts (24-hour, 48-hour, 7-day) with 90% confidence intervals.
        """
        records = []
        start_hr = int(base_input["hr"].iloc[0]) if "hr" in base_input.columns else 8
        base_dict = base_input.iloc[0].to_dict()

        for step in range(hours_ahead):
            current_hr = (start_hr + step) % 24
            curr_dict = base_dict.copy()
            curr_dict["hr"] = current_hr

            step_df = pd.DataFrame([curr_dict])
            pred_res = self.predict_single(step_df)

            records.append(
                {
                    "step_hour_ahead": step + 1,
                    "hour_of_day": current_hr,
                    "predicted_demand": pred_res["predicted_demand"],
                    "q10_bound": pred_res["q10_demand_bound"],
                    "q90_bound": pred_res["q90_demand_bound"],
                    "estimated_revenue_usd": pred_res["estimated_revenue_usd"],
                    "fleet_utilization_pct": pred_res["fleet_utilization_pct"],
                    "stockout_risk": pred_res["stockout_risk_level"],
                }
            )

        return pd.DataFrame(records)


forecaster = DemandForecaster()
