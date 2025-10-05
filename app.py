import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_exoplanet_model.pkl")

# Feature info: label, min, max, default
feature_info = {
    'kepid': {"label": "Kepler ID", "min": 1, "max": 10000000, "default": 100000},
    'koi_score': {"label": "Kepler Object Score", "min": 0, "max": 1, "default": 0.5},
    'koi_fpflag_nt': {"label": "Not Transit-Like Flag (0/1)", "min": 0, "max": 1, "default": 0},
    'koi_fpflag_ss': {"label": "Single Star Flag (0/1)", "min": 0, "max": 1, "default": 0},
    'koi_fpflag_co': {"label": "Centroid Offset Flag (0/1)", "min": 0, "max": 1, "default": 0},
    'koi_fpflag_ec': {"label": "Eclipsing Binary Flag (0/1)", "min": 0, "max": 1, "default": 0},
    'koi_period': {"label": "Orbital Period (days)", "min": 0.1, "max": 500, "default": 10},
    'koi_period_err1': {"label": "Orbital Period Error (+)", "min": 0, "max": 10, "default": 0.01},
    'koi_period_err2': {"label": "Orbital Period Error (-)", "min": 0, "max": 10, "default": 0.01},
    'koi_time0bk': {"label": "Transit Time (BKJD)", "min": 0, "max": 2000, "default": 1000},
    'koi_time0bk_err1': {"label": "Transit Time Error (+)", "min": 0, "max": 1, "default": 0.01},
    'koi_time0bk_err2': {"label": "Transit Time Error (-)", "min": 0, "max": 1, "default": 0.01},
    'koi_impact': {"label": "Impact Parameter", "min": 0, "max": 1, "default": 0.5},
    'koi_impact_err1': {"label": "Impact Parameter Error (+)", "min": 0, "max": 1, "default": 0.01},
    'koi_impact_err2': {"label": "Impact Parameter Error (-)", "min": 0, "max": 1, "default": 0.01},
    'koi_duration': {"label": "Transit Duration (hrs)", "min": 0, "max": 20, "default": 2.5},
    'koi_duration_err1': {"label": "Transit Duration Error (+)", "min": 0, "max": 10, "default": 0.1},
    'koi_duration_err2': {"label": "Transit Duration Error (-)", "min": 0, "max": 10, "default": 0.1},
    'koi_depth': {"label": "Transit Depth", "min": 0, "max": 0.1, "default": 0.001},
    'koi_depth_err1': {"label": "Transit Depth Error (+)", "min": 0, "max": 0.1, "default": 0.001},
    'koi_depth_err2': {"label": "Transit Depth Error (-)", "min": 0, "max": 0.1, "default": 0.001},
    'koi_prad': {"label": "Planet Radius (Earth radii)", "min": 0.1, "max": 50, "default": 1},
    'koi_prad_err1': {"label": "Planet Radius Error (+)", "min": 0, "max": 10, "default": 0.1},
    'koi_prad_err2': {"label": "Planet Radius Error (-)", "min": 0, "max": 10, "default": 0.1},
    'koi_teq': {"label": "Equilibrium Temperature (K)", "min": 0, "max": 5000, "default": 500},
    'koi_teq_err1': {"label": "Teq Error (+)", "min": 0, "max": 1000, "default": 50},
    'koi_teq_err2': {"label": "Teq Error (-)", "min": 0, "max": 1000, "default": 50},
    'koi_insol': {"label": "Insolation Flux (Earth flux)", "min": 0, "max": 1000, "default": 100},
    'koi_insol_err1': {"label": "Insolation Error (+)", "min": 0, "max": 500, "default": 10},
    'koi_insol_err2': {"label": "Insolation Error (-)", "min": 0, "max": 500, "default": 10},
    'koi_model_snr': {"label": "Model SNR", "min": 0, "max": 50, "default": 10},
    'koi_tce_plnt_num': {"label": "TCE Planet Number", "min": 1, "max": 10, "default": 1},
    'koi_steff': {"label": "Stellar Effective Temp (K)", "min": 2000, "max": 10000, "default": 5500},
    'koi_steff_err1': {"label": "Stellar Temp Error (+)", "min": 0, "max": 1000, "default": 50},
    'koi_steff_err2': {"label": "Stellar Temp Error (-)", "min": 0, "max": 1000, "default": 50},
    'koi_slogg': {"label": "Stellar Surface Gravity (log g)", "min": 0, "max": 5, "default": 4.0},
    'koi_slogg_err1': {"label": "Surface Gravity Error (+)", "min": 0, "max": 2, "default": 0.1},
    'koi_slogg_err2': {"label": "Surface Gravity Error (-)", "min": 0, "max": 2, "default": 0.1},
    'koi_srad': {"label": "Stellar Radius (R☉)", "min": 0.1, "max": 10, "default": 1},
    'koi_srad_err1': {"label": "Stellar Radius Error (+)", "min": 0, "max": 5, "default": 0.1},
    'koi_srad_err2': {"label": "Stellar Radius Error (-)", "min": 0, "max": 5, "default": 0.1},
    'ra': {"label": "RA (deg)", "min": 0, "max": 360, "default": 100},
    'dec': {"label": "DEC (deg)", "min": -90, "max": 90, "default": 20},
    'koi_kepmag': {"label": "Kepler Magnitude", "min": 8, "max": 20, "default": 13},
}

st.markdown("""
### 🚀 About
This AI/ML model predicts the class of an exoplanet (Candidate, Confirmed, or False Positive) using NASA's Kepler Objects of Interest dataset.

It uses advanced ensemble learning (Random Forest, XGBoost, LightGBM) to analyze features like orbital period, planet radius, stellar properties, and more.
""")


st.title("NASA Exoplanet AI/ML Predictor 🌌")
st.write("Upload a CSV or enter values manually to predict exoplanet class.")

# Mapping for predictions
class_mapping = {
    0: "CANDIDATE",
    1: "CONFIRMED",
    2: "FALSE POSITIVE"
}

class_description = {
    0: "Looks like a planet but not fully confirmed yet. Needs more observations.",
    1: "Verified as a real exoplanet through multiple observations.",
    2: "Not a planet, likely caused by noise, binary star, or instrument error."
}

# CSV upload
required_columns = list(feature_info.keys())

st.info(f"⚠️ Note: If you upload a CSV file, it must contain the following columns exactly (case-sensitive and in any order):\n\n{required_columns}")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Check if all required columns are present
    missing_cols = [col for col in required_columns if col not in input_df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from your CSV: {missing_cols}")
    else:
        st.success("All required columns found! Proceeding with prediction...")
        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)
        st.write("Predictions:", preds)
        st.dataframe(pd.DataFrame(probs, columns=[class_mapping[c] for c in model.classes_]))



# Manual input
else:
    st.subheader("Manual Input")
    input_data = {}
    for feature, info in feature_info.items():
        # Ensure float type for Streamlit number_input
        input_data[feature] = st.number_input(
            info["label"],
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["default"]),
            format="%.5f"
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]      # predicted class
        prob = model.predict_proba(input_df)[0]  # probabilities for all classes

        # Build DataFrame for all classes
        df = pd.DataFrame({
            "Class": [class_mapping[c] for c in model.classes_],
            "Explanation": [class_description[c] for c in model.classes_],
            "Probability": [f"{p*100:.2f}%" for p in prob]


        })

        st.subheader("Prediction Overview")
        st.dataframe(df)


with open("best_exoplanet_model.pkl", "rb") as f:
    st.download_button(
        label="⬇️ Download Trained Model (PKL)",
        data=f,
        file_name="best_exoplanet_model.pkl",
        mime="application/octet-stream"
    )

st.markdown("""
---
👩‍🚀 **Developed for NASA Space Apps Challenge 2025**
""")


