import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from configs.settings import settings
from src.mlops.prediction_logger import prediction_logger
from src.mlops.registry import model_registry
from src.models.forecaster import forecaster
from src.models.train_pipeline import get_bike_data, run_training_pipeline
from src.monitoring.drift_detector import drift_detector
from src.monitoring.system_metrics import SystemMetricsMonitor


def ensure_champion_model():
    try:
        model_registry.load_champion()
    except Exception:
        run_training_pipeline()


ensure_champion_model()

st.set_page_config(
    page_title="🚲 Enterprise Bike Rental Demand Platform",
    layout="wide",
    page_icon="🚲",
)

# Custom Executive Dark Theme
st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #38ef7d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .disclaimer-text {
        font-size: 0.75rem;
        color: #6e7681;
        font-style: italic;
        margin-top: 4px;
    }
    .badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-mod {
        background-color: #ffa726;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #38ef7d;
        color: #000;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚲 Enterprise Bike Rental Demand Forecasting & Fleet Optimization Platform")
st.caption(
    "Commercial-Grade ML Platform | Quantile Uncertainty | Side-by-Side Scenario Simulation | MLOps Drift & Audit"
)
st.markdown("---")

metadata = model_registry.load_metadata()

# 8-Tab Architecture
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "📊 Executive Dashboard",
        "🎯 Demand Prediction",
        "📤 Batch Prediction",
        "📈 Multi-Horizon Forecast",
        "⚖️ Scenario Comparison",
        "🔍 Model Explainability",
        "🛡️ MLOps & Drift Monitoring",
        "ℹ️ Platform Architecture",
    ]
)

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab1:
    st.header("Executive Operational Command Center")

    col1, col2, col3, col4 = st.columns(4)

    default_input = pd.DataFrame(
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

    res = forecaster.predict_single(default_input)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Peak Demand Forecast</div>
            <div class="metric-value">{res['predicted_demand']} <span style="font-size:1.1rem;color:#38ef7d;">bikes/hr</span></div>
            <div class="disclaimer-text">{res['disclaimer']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card" style="border-left-color: #00d2ff;">
            <div class="metric-label">90% Confidence Interval</div>
            <div class="metric-value">{res['q10_demand_bound']} – {res['q90_demand_bound']}</div>
            <div class="disclaimer-text">Quantile Regression Bounds</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card" style="border-left-color: #ab47bc;">
            <div class="metric-label">Estimated Hourly Revenue</div>
            <div class="metric-value">${res['estimated_revenue_usd']:,.2f}</div>
            <div class="disclaimer-text">{res['disclaimer']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        risk = res["stockout_risk_level"]
        badge_cls = (
            "badge-high"
            if risk == "High"
            else ("badge-mod" if risk == "Moderate" else "badge-low")
        )
        st.markdown(
            f"""
        <div class="metric-card" style="border-left-color: #ffa726;">
            <div class="metric-label">Fleet Utilization & Stockout Risk</div>
            <div class="metric-value" style="font-size:1.6rem;">{res['fleet_utilization_pct']}% <span class="{badge_cls}">{risk}</span></div>
            <div class="disclaimer-text">{res['disclaimer']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("📊 Quantile Demand Uncertainty Bounds")
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                name="P10 Lower Bound",
                x=["Demand Bounds"],
                y=[res["q10_demand_bound"]],
                marker_color="#00c6ff",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name="P50 Median Forecast",
                x=["Demand Bounds"],
                y=[res["predicted_demand"] - res["q10_demand_bound"]],
                marker_color="#38ef7d",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name="P90 Upper Bound",
                x=["Demand Bounds"],
                y=[res["q90_demand_bound"] - res["predicted_demand"]],
                marker_color="#ff007f",
            )
        )
        fig_bar.update_layout(
            barmode="stack",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            showlegend=True,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("📋 Operational Directives")
        for rec in res["operational_recommendations"]:
            st.info(rec)
        st.caption(res["disclaimer"])

# ==========================================
# TAB 2: DEMAND PREDICTION
# ==========================================
with tab2:
    st.header("Single Scenario Forecast & Rebalancing Engine")

    with st.form("scenario_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            season = st.selectbox(
                "Season", ["springer", "summer", "fall", "winter"], index=1
            )
            weather = st.selectbox(
                "Weather Condition",
                ["Clear", "Mist", "Light Snow", "Heavy Rain"],
                index=0,
            )
            hr = st.number_input("Hour of Day (0-23)", 0, 23, 18)
        with col_b:
            temp = st.slider("Normalized Temp (0.0 to 1.0)", 0.0, 1.0, 0.65)
            hum = st.slider("Normalized Humidity (0.0 to 1.0)", 0.0, 1.0, 0.45)
            windspeed = st.slider("Normalized Windspeed (0.0 to 1.0)", 0.0, 1.0, 0.15)
        with col_c:
            yr = st.selectbox("Year", ["2011", "2012"], index=1)
            mnth = st.number_input("Month (1-12)", 1, 12, 6)
            weekday = st.number_input("Weekday (0=Sun, 6=Sat)", 0, 6, 2)
            holiday = st.selectbox("Holiday?", ["No", "Yes"], index=0)
            workingday = st.selectbox(
                "Working Day?", ["Working Day", "No work"], index=0
            )

        submit_btn = st.form_submit_button("🚀 Compute Prediction & Logistics")

    scen_df = pd.DataFrame(
        [
            {
                "season": season,
                "yr": yr,
                "mnth": float(mnth),
                "hr": int(hr),
                "holiday": holiday,
                "weekday": int(weekday),
                "workingday": workingday,
                "weathersit": weather,
                "temp": float(temp),
                "hum": float(hum),
                "windspeed": float(windspeed),
            }
        ]
    )

    scen_res = forecaster.predict_single(scen_df)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Predicted Demand", f"{scen_res['predicted_demand']} bikes")
    p2.metric(
        "90% CI Bounds",
        f"{scen_res['q10_demand_bound']} – {scen_res['q90_demand_bound']}",
    )
    p3.metric("Rebalancing Trucks", f"{scen_res['trucks_recommended']} trucks")
    p4.metric("Staff Allocation", f"{scen_res['staff_recommended']} staff")
    st.caption(scen_res["disclaimer"])

# ==========================================
# TAB 3: BATCH PREDICTION
# ==========================================
with tab3:
    st.header("Batch CSV Bulk Inference Engine")
    st.markdown(
        "Upload schedule CSV to compute bulk forecasts, 90% confidence bounds, and rebalancing requirements."
    )

    file_up = st.file_uploader("Upload CSV File", type=["csv"])
    if file_up is not None:
        batch_raw = pd.read_csv(file_up)
        st.dataframe(batch_raw.head(5))

        if st.button("Run Bulk Inference"):
            predictions = []
            for _, row in batch_raw.iterrows():
                row_df = pd.DataFrame([row.to_dict()])
                r = forecaster.predict_single(row_df)
                predictions.append(
                    {
                        "Predicted_Demand": r["predicted_demand"],
                        "P10_Bound": r["q10_demand_bound"],
                        "P90_Bound": r["q90_demand_bound"],
                        "Estimated_Revenue_USD": r["estimated_revenue_usd"],
                        "Fleet_Utilization_PCT": r["fleet_utilization_pct"],
                        "Trucks_Needed": r["trucks_recommended"],
                        "Stockout_Risk": r["stockout_risk_level"],
                    }
                )

            pred_df = pd.concat([batch_raw, pd.DataFrame(predictions)], axis=1)
            st.success(f"Computed bulk predictions for {len(pred_df)} records!")
            st.dataframe(pred_df)

            csv_data = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Bulk Predictions CSV",
                data=csv_data,
                file_name="bike_demand_batch_predictions.csv",
                mime="text/csv",
            )
            st.caption("Calculated using configurable business assumptions.")

# ==========================================
# TAB 4: MULTI-HORIZON FORECASTING
# ==========================================
with tab4:
    st.header("Multi-Horizon Trend Forecasting (24h, 48h, 7-Day)")

    horizon_opt = st.selectbox(
        "Select Forecast Horizon",
        ["24-Hour Forecast", "48-Hour Forecast", "7-Day Forecast (168h)"],
    )
    h_hours = 24 if "24" in horizon_opt else (48 if "48" in horizon_opt else 168)

    forecast_df = forecaster.predict_multi_horizon(default_input, hours_ahead=h_hours)

    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df["step_hour_ahead"],
            y=forecast_df["predicted_demand"],
            mode="lines+markers",
            name="Median Forecast (P50)",
            line=dict(color="#38ef7d", width=3),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df["step_hour_ahead"],
            y=forecast_df["q90_bound"],
            mode="lines",
            name="Upper Bound (P90)",
            line=dict(color="#ff007f", dash="dot"),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df["step_hour_ahead"],
            y=forecast_df["q10_bound"],
            mode="lines",
            name="Lower Bound (P10)",
            fill="tonexty",
            fillcolor="rgba(56, 239, 125, 0.15)",
            line=dict(color="#00c6ff", dash="dot"),
        )
    )
    fig_fc.update_layout(
        title=f"Demand Horizon Forecast ({h_hours} Hours Ahead)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Step Hours Ahead",
        yaxis_title="Predicted Demand (Bikes)",
        height=450,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# ==========================================
# TAB 5: SCENARIO COMPARISON (SIDE-BY-SIDE)
# ==========================================
with tab5:
    st.header("⚖️ Side-by-Side Dual Scenario Simulator")
    st.markdown(
        "Compare two weather or operational scenarios side-by-side to analyze demand and financial impact."
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("Scenario A (Baseline)")
        weather_a = st.selectbox(
            "Weather A",
            ["Clear", "Mist", "Light Snow", "Heavy Rain"],
            index=0,
            key="w_a",
        )
        temp_a = st.slider("Temp A (0.0 - 1.0)", 0.0, 1.0, 0.65, key="t_a")
        hr_a = st.slider("Hour A (0 - 23)", 0, 23, 18, key="h_a")
        work_a = st.selectbox(
            "Workday A", ["Working Day", "No work"], index=0, key="wk_a"
        )

        df_a = pd.DataFrame(
            [
                {
                    "season": "summer",
                    "yr": "2012",
                    "mnth": 6.0,
                    "hr": hr_a,
                    "holiday": "No",
                    "weekday": 2,
                    "workingday": work_a,
                    "weathersit": weather_a,
                    "temp": temp_a,
                    "hum": 0.45,
                    "windspeed": 0.15,
                }
            ]
        )
        res_a = forecaster.predict_single(df_a)

        st.metric("Demand A", f"{res_a['predicted_demand']} bikes")
        st.metric("Revenue A", f"${res_a['estimated_revenue_usd']:,.2f}")
        st.metric("Trucks A", f"{res_a['trucks_recommended']} trucks")

    with col_s2:
        st.subheader("Scenario B (Comparison)")
        weather_b = st.selectbox(
            "Weather B",
            ["Clear", "Mist", "Light Snow", "Heavy Rain"],
            index=3,
            key="w_b",
        )
        temp_b = st.slider("Temp B (0.0 - 1.0)", 0.0, 1.0, 0.30, key="t_b")
        hr_b = st.slider("Hour B (0 - 23)", 0, 23, 18, key="h_b")
        work_b = st.selectbox(
            "Workday B", ["Working Day", "No work"], index=0, key="wk_b"
        )

        df_b = pd.DataFrame(
            [
                {
                    "season": "summer",
                    "yr": "2012",
                    "mnth": 6.0,
                    "hr": hr_b,
                    "holiday": "No",
                    "weekday": 2,
                    "workingday": work_b,
                    "weathersit": weather_b,
                    "temp": temp_b,
                    "hum": 0.85,
                    "windspeed": 0.35,
                }
            ]
        )
        res_b = forecaster.predict_single(df_b)

        demand_delta = res_b["predicted_demand"] - res_a["predicted_demand"]
        revenue_delta = res_b["estimated_revenue_usd"] - res_a["estimated_revenue_usd"]
        truck_delta = res_b["trucks_recommended"] - res_a["trucks_recommended"]

        st.metric(
            "Demand B",
            f"{res_b['predicted_demand']} bikes",
            delta=f"{demand_delta} bikes",
        )
        st.metric(
            "Revenue B",
            f"${res_b['estimated_revenue_usd']:,.2f}",
            delta=f"${revenue_delta:,.2f}",
        )
        st.metric(
            "Trucks B",
            f"{res_b['trucks_recommended']} trucks",
            delta=f"{truck_delta} trucks",
        )

    st.caption(
        "Business outputs are calculated using configurable assumptions and should be interpreted as planning estimates."
    )

# ==========================================
# TAB 6: MODEL EXPLAINABILITY
# ==========================================
with tab6:
    st.header("SHAP & Model Diagnostic Visualizations")

    exp_tabs = st.tabs(
        [
            "SHAP Summary",
            "Feature Importances",
            "Residual Diagnostics",
            "Prediction vs Actual",
        ]
    )

    fig_dir = settings.figures_dir
    with exp_tabs[0]:
        if (fig_dir / "shap_summary.png").exists():
            st.image(Image.open(fig_dir / "shap_summary.png"), use_container_width=True)
        if (fig_dir / "shap_dependence.png").exists():
            st.image(
                Image.open(fig_dir / "shap_dependence.png"), use_container_width=True
            )

    with exp_tabs[1]:
        if (fig_dir / "feature_importance.png").exists():
            st.image(
                Image.open(fig_dir / "feature_importance.png"), use_container_width=True
            )

    with exp_tabs[2]:
        if (fig_dir / "residual_distribution.png").exists():
            st.image(
                Image.open(fig_dir / "residual_distribution.png"),
                use_container_width=True,
            )

    with exp_tabs[3]:
        if (fig_dir / "prediction_vs_actual.png").exists():
            st.image(
                Image.open(fig_dir / "prediction_vs_actual.png"),
                use_container_width=True,
            )

# ==========================================
# TAB 7: MLOPS & DRIFT MONITORING
# ==========================================
with tab7:
    st.header("Evidently AI Drift & System Hardware Monitoring")

    m_col1, m_col2, m_col3 = st.columns(3)
    sys_health = SystemMetricsMonitor.get_system_health()

    with m_col1:
        st.metric("CPU Utilization", f"{sys_health['cpu_utilization_pct']}%")
    with m_col2:
        st.metric("RAM Used", f"{sys_health['memory_used_mb']} MB")
    with m_col3:
        st.metric("RAM Utilization", f"{sys_health['memory_utilization_pct']}%")

    st.markdown("---")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.subheader("🏆 Model Registry Metadata")
        st.json(metadata)

    with d_col2:
        st.subheader("🛡️ Drift Analysis Summary")
        if st.button("Run Evidently Drift Check"):
            ref_df = get_bike_data()
            curr_df = ref_df.sample(min(200, len(ref_df)), random_state=42)
            d_res = drift_detector.run_drift_analysis(ref_df, curr_df)
            st.success("Drift analysis complete!")
            st.json(d_res)

    st.markdown("---")
    st.subheader("📜 Recent Prediction Audit Trail (SQLite)")
    audit_df = prediction_logger.get_recent_predictions(limit=15)
    if not audit_df.empty:
        st.dataframe(audit_df)
    else:
        st.info("No inference prediction records logged yet.")

# ==========================================
# TAB 8: ABOUT PLATFORM
# ==========================================
with tab8:
    st.header("Enterprise Platform Architecture & Design")
    st.markdown("""
    ### 🏗️ Production System Stack
    - **Machine Learning Core**: Scikit-Learn, XGBoost, LightGBM, CatBoost, Optuna
    - **Centralized Feature Store**: `src/features/feature_store.py`
    - **Data Validation & Schemas**: Pandera schema contracts
    - **Experiment Tracking**: Local MLflow tracking (`./mlruns`)
    - **REST API Microservice**: FastAPI, SlowAPI Rate Limiting, Pydantic, Loguru
    - **Monitoring & Audit**: Evidently AI drift detection & SQLite prediction audit logging
    - **Container Orchestration**: Docker, Docker Compose, GitHub Actions CI/CD
    """)
