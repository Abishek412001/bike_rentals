import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the model bundle
bundle = joblib.load("bike_demand_model_bundle.pkl")
model = bundle["model"]
pipeline = bundle["pipeline"]

st.set_page_config(page_title="Bike Demand Predictor", layout="wide")

st.title("🚲 Bike Rental Demand Prediction Engine")
st.markdown("---")

# SIDEBAR: Environmental Settings
st.sidebar.header("Environmental Conditions")
season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
temp = st.sidebar.slider("Temperature (Normalized 0-1)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity (Normalized 0-1)", 0.0, 1.0, 0.5)
wind = st.sidebar.slider("Windspeed (Normalized 0-1)", 0.0, 1.0, 0.1)

# MAIN FORM: Temporal Settings
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        hr = st.number_input("Hour of Day (0-23)", 0, 23, 8)
        yr = st.selectbox("Year", ["2011", "2012"])
    with col2:
        mnth = st.number_input("Month (1-12)", 1, 12, 1)
        weekday = st.number_input("Weekday (0=Sun, 6=Sat)", 0, 6, 1)
    with col3:
        holiday = st.selectbox("Holiday?", ["No", "Yes"])
        workingday = st.selectbox("Working Day?", ["Working Day", "No work"])

    predict_btn = st.form_submit_button("Predict Demand")

if predict_btn:
    # 1. Feature Engineering (Match the notebook logic)
    is_peak = 1 if (7 <= hr <= 9) or (17 <= hr <= 19) else 0
    is_work = 1 if workingday == "Working Day" else 0
    commute_interaction = is_peak * is_work

    input_data = pd.DataFrame(
        {
            "season": [season],
            "yr": [yr],
            "mnth": [float(mnth)],
            "hr": [hr],
            "holiday": [holiday],
            "weekday": [weekday],
            "workingday": [workingday],
            "weathersit": [weather],
            "temp": [temp],
            "hum": [hum],
            "windspeed": [wind],
            "windspeed_winsorized": [wind],  # Approx for app
            "is_peak": [is_peak],
            "commute_interaction": [commute_interaction],
            "temp_sq": [temp**2],
            "hr_sin": [np.sin(2 * np.pi * hr / 24)],
            "hr_cos": [np.cos(2 * np.pi * hr / 24)],
            "hum_bins": ["Ideal"],  # Default label for simplicity in demo
        }
    )

    # 2. Inference
    prepared_data = pipeline.transform(input_data)
    pred_log = model.predict(prepared_data)
    prediction = np.expm1(pred_log)[0]

    # 3. Display
    st.success(f"### Predicted Demand: {int(prediction)} bikes")

    # 4. Explanation Logic
    st.info("**Operational Context:**")
    if is_peak and is_work:
        st.write(
            "⚠️ **Commute Peak Detected:** Demand is driven by regular transit users. Ensure high availability."
        )
    if temp > 0.8:
        st.write(
            "🌡️ **High Heat Warning:** Extreme temps may suppress demand below normal peaks."
        )
    if weather in ["Light Snow", "Heavy Rain"]:
        st.write(
            "🌧️ **Weather Suppression:** Demand is significantly lower due to precipitation."
        )

st.markdown("---")
st.header("📊 Project Insights & Analytics")

tab1, tab2, tab3 = st.tabs(["Performance", "Heatmap", "Explainability"])

with tab1:
    st.subheader("Model Accuracy: Actual vs Predicted")
    if Path("reports/figures/prediction_vs_actual.png").exists():
        st.image("reports/figures/prediction_vs_actual.png", use_container_width=True)

with tab2:
    st.subheader("Feature Correlation Heatmap")
    if Path("reports/figures/correlation_heatmap.png").exists():
        st.image("reports/figures/correlation_heatmap.png", use_container_width=True)

with tab3:
    st.subheader("Global Feature Importance")
    if Path("reports/figures/feature_importance.png").exists():
        st.image("reports/figures/feature_importance.png", use_container_width=True)
