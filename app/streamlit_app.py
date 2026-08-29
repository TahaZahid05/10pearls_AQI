import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import dagshub
import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CITY_NAME, FORECAST_HORIZONS
from src.data_fetcher import fetch_combined_data
from src.feature_engineering import engineer_features

DAGSHUB_OWNER = "TahaZahid05"
DAGSHUB_REPO = "10pearls_AQI"
MODEL_REGISTRY_NAME = "AQI_Predictor_Model"

st.set_page_config(
    page_title=f"{CITY_NAME} Air Quality Forecast",
    layout="wide",
)


def get_aqi_category(aqi_val: float):
    if aqi_val <= 50:
        return "Good", "#00e400", "#000000"
    elif aqi_val <= 100:
        return "Moderate", "#ffff00", "#000000"
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "#ffffff"
    elif aqi_val <= 200:
        return "Unhealthy", "#ff0000", "#ffffff"
    elif aqi_val <= 300:
        return "Very Unhealthy", "#8f3f97", "#ffffff"
    else:
        return "Hazardous", "#7e0023", "#ffffff"


@st.cache_data(ttl=1800)
def load_live_data():
    df_raw = fetch_combined_data(past_days=3, forecast_days=1)
    df_features = engineer_features(df_raw, is_training=False)
    return df_raw, df_features


@st.cache_resource
def load_model_bundle():
    try:
        dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
        client = mlflow.tracking.MlflowClient()
        reg_model = client.get_registered_model(MODEL_REGISTRY_NAME)
        latest_version = reg_model.latest_versions[-1]
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=latest_version.run_id, artifact_path="model/best_model.pkl"
        )
        bundle = joblib.load(artifact_path)
        bundle["source"] = f"DagsHub Model Registry (v{latest_version.version})"
        return bundle
    except Exception:
        local_path = BASE_DIR / "models" / "best_model.pkl"
        if local_path.exists():
            bundle = joblib.load(local_path)
            bundle["source"] = "Local Artifact (models/best_model.pkl)"
            return bundle
    return None


def run_inference(bundle, df_features):
    feature_cols = bundle["feature_cols"]
    latest_row = df_features[feature_cols].iloc[[-1]]
    latest_time = df_features["time"].iloc[-1]

    predictions = {}
    for h in FORECAST_HORIZONS:
        target_col = f"target_aqi_{h}h"
        model = bundle["models"][target_col]
        pred_val = model.predict(latest_row)[0]
        pred_time = latest_time + pd.Timedelta(hours=h)
        predictions[h] = {
            "val": float(pred_val),
            "time": pred_time,
            "day": h // 24,
        }
    return predictions, latest_row.iloc[0].to_dict(), latest_time


def main():
    bundle = load_model_bundle()
    if bundle is None:
        st.error("Model artifact not found in DagsHub Model Store or local cache.")
        return

    st.title(f"{CITY_NAME} Air Quality Index (AQI) - 3-Day Forecast")
    st.caption(f"Real-time forecasting powered by LightGBM • Loaded from {bundle.get('source', 'Model Store')}")

    with st.spinner("Fetching latest live data for Karachi..."):
        df_raw, df_features = load_live_data()

    if len(df_features) == 0:
        st.error("Unable to compute feature lags from live data stream.")
        return

    predictions, latest_features, latest_time = run_inference(bundle, df_features)
    current_aqi = df_features["us_aqi"].iloc[-1]
    current_cat, current_bg, current_fg = get_aqi_category(current_aqi)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: {current_bg}; color: {current_fg}; padding: 18px; border-radius: 8px; text-align: center;">
                <h4 style="margin: 0; font-size: 14px; text-transform: uppercase;">Current Live AQI</h4>
                <h1 style="margin: 5px 0; font-size: 42px;">{int(round(current_aqi))}</h1>
                <p style="margin: 0; font-weight: bold;">{current_cat}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Latest Observation Time", latest_time.strftime("%Y-%m-%d %H:%M"))
        st.metric("Dominant Pollutant (PM2.5)", f"{latest_features.get('pm2_5', 0):.1f} µg/m³")

    with col3:
        st.metric("Temperature", f"{latest_features.get('temperature_2m', 0):.1f} °C")
        st.metric("Relative Humidity", f"{latest_features.get('relative_humidity_2m', 0):.1f} %")

    st.markdown("---")

    st.subheader("3-Day Forecast Predictions")
    f_cols = st.columns(3)

    max_pred = 0
    for idx, h in enumerate(FORECAST_HORIZONS):
        pred_info = predictions[h]
        pred_val = pred_info["val"]
        pred_time = pred_info["time"]
        day_num = pred_info["day"]
        cat, bg, fg = get_aqi_category(pred_val)
        max_pred = max(max_pred, pred_val)

        with f_cols[idx]:
            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; border-top: 6px solid {bg}; padding: 15px; border-radius: 6px; background-color: #fafafa;">
                    <h4 style="margin: 0; color: #333;">Day {day_num} (+{h}h)</h4>
                    <p style="margin: 2px 0 8px 0; color: #777; font-size: 12px;">{pred_time.strftime('%a, %d %b %H:%M')}</p>
                    <h2 style="margin: 0; color: #111;">{int(round(pred_val))} AQI</h2>
                    <span style="display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 4px; font-size: 12px; background: {bg}; color: {fg}; font-weight: bold;">
                        {cat}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if max_pred > 150:
        st.error(
            "Health Alert: Air quality is forecasted to reach Unhealthy levels (>150 AQI). "
            "Active children and adults, and people with respiratory disease, such as asthma, should avoid prolonged outdoor exertion."
        )
    elif max_pred > 100:
        st.warning(
            "Air Quality Warning: Air quality is forecasted to be Unhealthy for Sensitive Groups (>100 AQI). "
            "People with respiratory or heart disease, the elderly, and children should limit prolonged outdoor exertion."
        )
    else:
        st.info("Air quality is forecasted to remain within acceptable levels over the next 3 days.")

    st.subheader("AQI Historical Trajectory & 3-Day Forecast Curve")

    hist_df = df_features[["time", "us_aqi"]].tail(72)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["time"],
            y=hist_df["us_aqi"],
            mode="lines+markers",
            name="Past 72h Actual AQI",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=4),
        )
    )

    forecast_times = [latest_time] + [predictions[h]["time"] for h in FORECAST_HORIZONS]
    forecast_vals = [current_aqi] + [predictions[h]["val"] for h in FORECAST_HORIZONS]

    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=forecast_vals,
            mode="lines+markers",
            name="3-Day Predicted AQI",
            line=dict(color="#e65100", width=3, dash="dash"),
            marker=dict(size=8, color="#e65100"),
        )
    )

    thresholds = [
        (50, "Good (50)", "green"),
        (100, "Moderate (100)", "gold"),
        (150, "Unhealthy for Sensitive (150)", "orange"),
        (200, "Unhealthy (200)", "red"),
    ]
    for val, label, color in thresholds:
        fig.add_hline(
            y=val,
            line_dash="dot",
            line_color=color,
            annotation_text=label,
            annotation_position="bottom right",
            opacity=0.6,
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="US AQI",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current Atmospheric & Pollutant Readings")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric("PM10", f"{latest_features.get('pm10', 0):.1f} µg/m³")
        st.metric("Carbon Monoxide", f"{latest_features.get('carbon_monoxide', 0):.1f} µg/m³")

    with m_col2:
        st.metric("Nitrogen Dioxide", f"{latest_features.get('nitrogen_dioxide', 0):.1f} µg/m³")
        st.metric("Sulphur Dioxide", f"{latest_features.get('sulphur_dioxide', 0):.1f} µg/m³")

    with m_col3:
        st.metric("Ozone", f"{latest_features.get('ozone', 0):.1f} µg/m³")
        st.metric("Surface Pressure", f"{latest_features.get('surface_pressure', 0):.1f} hPa")

    with m_col4:
        st.metric("Wind Speed", f"{latest_features.get('wind_speed_10m', 0):.1f} km/h")
        st.metric("Wind Direction", f"{latest_features.get('wind_direction_10m', 0):.0f}°")

    st.markdown("---")

    st.subheader("Model Explainability (SHAP Feature Importance)")
    st.write(
        "The chart below illustrates the atmospheric drivers impacting the model's 24-hour predictions. "
        "Positive SHAP values push predicted AQI higher, while negative values push it lower."
    )

    shap_path = BASE_DIR / "plots" / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Summary (Top Drivers of Karachi AQI)", use_container_width=True)
    else:
        st.info("SHAP plot artifact not found. Run training script to generate SHAP visualizations.")


if __name__ == "__main__":
    main()
