from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breathe_smart_sample.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="BreatheSmart", page_icon="🌿", layout="wide")
st.title("🌿 BreatheSmart")
st.subheader("Air Quality & Pollution-Exposure Risk Explorer")
st.caption("Machine-learning demonstration • Synthetic sample data • Not a medical device")

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temperature", "humidity"]
UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³",
    "CO": "mg/m³", "O3": "µg/m³", "temperature": "°C", "humidity": "%"
}

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if not (MODEL_DIR / "aqi_model.joblib").exists() or not (MODEL_DIR / "health_model.joblib").exists():
    st.error("Trained models are missing. In the project folder, run: python train_model.py")
    st.stop()
if not DATA_PATH.exists():
    st.error("Dataset is missing. Run: python generate_dataset.py")
    st.stop()

aqi_bundle = load_bundle(str(MODEL_DIR / "aqi_model.joblib"))
health_bundle = load_bundle(str(MODEL_DIR / "health_model.joblib"))
df = load_data(str(DATA_PATH))

with st.sidebar:
    st.header("Enter pollutant measurements")
    st.write("Use readings from a monitoring station when available.")
    values = {
        "PM2.5": st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 45.0, 1.0),
        "PM10": st.slider("PM10 (µg/m³)", 0.0, 500.0, 85.0, 1.0),
        "NO2": st.slider("NO₂ (µg/m³)", 0.0, 500.0, 40.0, 1.0),
        "SO2": st.slider("SO₂ (µg/m³)", 0.0, 500.0, 15.0, 1.0),
        "CO": st.slider("CO (mg/m³)", 0.0, 40.0, 1.0, 0.1),
        "O3": st.slider("O₃ (µg/m³)", 0.0, 500.0, 35.0, 1.0),
        "temperature": st.slider("Temperature (°C)", 0.0, 50.0, 28.0, 0.5),
        "humidity": st.slider("Relative humidity (%)", 0.0, 100.0, 55.0, 1.0),
    }
    
# Prediction button
predict_btn = st.button(
    "Predict air quality",
    type="primary",
    use_container_width=True
)

if predict_btn:
    # Prepare input from the sidebar values
    input_df = pd.DataFrame([values], columns=FEATURES)

    # Run both trained machine learning models
    aqi_pred = aqi_bundle["model"].predict(input_df)[0]
    health_pred = health_bundle["model"].predict(input_df)[0]

    st.success("Prediction completed successfully!")

    # Display prediction results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted air quality")
        st.metric("AQI category", str(aqi_pred))
        st.caption(
            f"Model: {aqi_bundle.get('model_name', 'Random Forest')}"
        )

    with col2:
        st.subheader("Estimated pollution-exposure risk")
        st.metric("Risk category", str(health_pred))
        st.caption(
            f"Model: {health_bundle.get('model_name', 'Random Forest')}"
        )

    # Show practical guidance
    advice = {
        "Good": "Air quality is generally favourable.",
        "Satisfactory": "Most people can continue usual activities.",
        "Moderate": "Sensitive individuals should consider reducing prolonged outdoor exertion.",
        "Poor": "Reduce prolonged outdoor exertion where possible.",
        "Very Poor": "Limit outdoor exposure and follow local air-quality advisories.",
        "Severe": "Minimise outdoor exposure and follow official local alerts."
    }

    st.subheader("Practical guidance")
    st.info(advice.get(str(aqi_pred), "Check official local air-quality guidance."))

else:
    st.info(
        "Adjust the pollutant measurements in the sidebar, "
        "then click Predict air quality to generate results."
    )


st.markdown("### Current input summary")
cols = st.columns(4)
for i, feature in enumerate(FEATURES):
    cols[i % 4].metric(feature, f"{values[feature]:g} {UNITS[feature]}")

input_df = pd.DataFrame([values], columns=FEATURES)
aqi_pred = aqi_bundle["model"].predict(input_df)[0]
health_pred = health_bundle["model"].predict(input_df)[0]

# Prediction is updated whenever the user changes a slider. The button is kept
# as a clear interaction cue; no external/live monitoring data is implied.
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Predicted air quality")
    st.metric("AQI category", str(aqi_pred))
    st.caption(f"Model: {aqi_bundle.get('model_name', 'trained classifier')}")
with col2:
    st.markdown("### Estimated pollution-exposure risk")
    st.metric("Risk category", str(health_pred))
    st.caption(f"Model: {health_bundle.get('model_name', 'trained classifier')}")

advice = {
    "Good": "Air quality is generally favourable. Continue normal activities while following local advisories.",
    "Satisfactory": "Most people can continue usual activities. Sensitive individuals should pay attention to symptoms.",
    "Moderate": "People with asthma or heart/lung conditions, children, and older adults should consider reducing prolonged strenuous outdoor activity.",
    "Poor": "Consider reducing prolonged outdoor exertion, particularly if you are sensitive to air pollution.",
    "Very Poor": "Avoid or reduce outdoor exertion where possible and follow local health advisories.",
    "Severe": "Minimise outdoor exposure and follow official local alerts. Seek medical advice for severe or persistent symptoms.",
}
st.markdown("### Practical guidance")
st.info(advice.get(str(aqi_pred), "Check official local air-quality guidance and limit exposure when pollution is elevated."))

st.markdown("### Explore the sample dataset")
st.caption("The bundled CSV is synthetic demonstration data and does not represent current air quality in any city.")
m1, m2, m3 = st.columns(3)
m1.metric("Sample records", f"{len(df):,}")
m2.metric("Input features", len(FEATURES))
m3.metric("AQI classes", df["AQI_Category"].nunique())

left, right = st.columns(2)
with left:
    st.markdown("**Air-quality category distribution**")
    counts = df["AQI_Category"].value_counts().reindex(
        ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"], fill_value=0
    )
    st.bar_chart(counts)
with right:
    st.markdown("**PM2.5 distribution**")
    hist = df["PM2.5"].round(0).value_counts().sort_index()
    st.line_chart(hist)

with st.expander("View sample records"):
    st.dataframe(df.head(100), use_container_width=True)

metrics_path = MODEL_DIR / "metrics.json"
if metrics_path.exists():
    with st.expander("Model evaluation (synthetic hold-out test set)"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for target, info in metrics.items():
            st.markdown(f"**{target} — selected model: {info['selected_model']}**")
            score_df = pd.DataFrame(info["scores"]).T
            st.dataframe(score_df, use_container_width=True)
        st.warning("These scores reflect synthetic data generated by rules similar to the labels. They are not evidence of real-world accuracy.")

st.markdown("---")
st.caption("Educational prototype only. This tool does not diagnose illness, calculate a person's medical risk, or replace official AQI readings or professional medical advice.")
