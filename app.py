# app.py — Streamlit front-end for first half
import numpy as np
import pandas as pd
import streamlit as st
import joblib, xgboost as xgb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="PPOP – Hemophilia Decision Support", layout="centered")
st.title("PPOP – Hemophilia Decision Support")

# ---------------- Utilities ----------------
def inv_feature_minmax(scaler, idx, v_scaled):
    # Inverse transform a single feature by index for MinMaxScaler
    return v_scaled * (scaler.data_max_[idx] - scaler.data_min_[idx]) + scaler.data_min_[idx]

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    preproc_xgb = joblib.load("preproc_xgb.pkl")
    xgb_booster = xgb.Booster(); xgb_booster.load_model("xgb_booster.json")
    preproc_rf  = joblib.load("preproc_rf.pkl")
    rf_model    = joblib.load("rf_model.pkl")
    with open("rf_threshold.txt") as f:
        best_thr = float(f.read().strip())
    lstm_model  = load_model("lstm_model.keras", compile=False)
    lstm_scaler = joblib.load("lstm_scaler.pkl")
    return preproc_xgb, xgb_booster, preproc_rf, rf_model, best_thr, lstm_model, lstm_scaler

preproc_xgb, xgb_booster, preproc_rf, rf_model, best_thr, lstm_model, lstm_scaler = load_artifacts()

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Inputs")
    age = st.slider("Age", 5, 65, 30)
    weight = st.slider("Weight (kg)", 20, 90, 60)
    dose = st.selectbox("Infusion Dose (IU)", [250, 500, 1000, 1500, 2000, 2500], index=4)
    time_since = st.slider("Time since last infusion (hr)", 0, 168, 24)
    severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"], index=2)
    horizon = st.slider("Prediction Horizon (hours)", 24, 168, 72, step=24)
    enforce_decay = st.checkbox("Enforce monotonic decay (no new infusion)", value=True)

if st.button("Predict"):
    # ---------------- Factor level prediction (XGB) ----------------
    row = {
        "age": age, "weight_kg": weight,
        "infusion_dose_IU": dose,
        "time_since_last_infusion_hr": time_since,
        "tsi_sq": time_since ** 2,
        "log_dose": np.log1p(dose),
        "dose_per_kg": dose / weight,
        "severity": severity
    }
    input_df = pd.DataFrame([row])
    dmat = xgb.DMatrix(preproc_xgb.transform(input_df))
    factor_now = float(xgb_booster.predict(dmat)[0])

    # ---------------- Bleed risk prediction (RF) ----------------
    row_rf = dict(row); row_rf["factor_level_IU_dL"] = factor_now
    X_proc_rf = preproc_rf.transform(pd.DataFrame([row_rf]))
    proba = float(rf_model.predict_proba(X_proc_rf)[:, 1][0])
    risk_label = "High" if proba >= best_thr else "Low"

    # ---------------- Temporal Prediction ----------------
    seq_features = ["infusion_dose_IU", "time_since_last_infusion_hr", "factor_level_IU_dL"]
    steps = horizon // 6  # every 6 hours
    preds, times = [factor_now], [0]

    if enforce_decay:
        # ---- Option A: Pure exponential decay ----
        decay_rate = 0.015  # tune
        for i in range(steps):
            pred_val = factor_now * np.exp(-decay_rate * (i+1) * 6)
            preds.append(pred_val)
            times.append((i+1) * 6)

    else:
        # ---- Option B: LSTM with feedback ----
        # Initialize with current state
        start_unscaled = np.array([[dose, time_since, factor_now]])
        start_scaled = lstm_scaler.transform(pd.DataFrame(start_unscaled, columns=seq_features))[0]
        seq_window = np.tile(start_scaled, (5, 1))
        current_seq = seq_window.reshape(1, seq_window.shape[0], seq_window.shape[1])

        last_pred = factor_now
        for i in range(steps):
            pred_scaled = lstm_model.predict(current_seq, verbose=0).flatten()[0]

            inv = lstm_scaler.inverse_transform(pd.DataFrame(
                [[dose, time_since + (i+1)*6, pred_scaled]], columns=seq_features))
            lstm_pred = inv[0, -1]

            preds.append(lstm_pred)
            times.append((i+1) * 6)
            last_pred = lstm_pred

            # update input sequence
            next_unscaled = np.array([[dose, time_since + (i+1)*6, last_pred]])
            next_scaled = lstm_scaler.transform(pd.DataFrame(next_unscaled, columns=seq_features))[0]
            current_seq = np.vstack([current_seq[0, 1:], next_scaled]).reshape(
                1, seq_window.shape[0], seq_window.shape[1]
            )

    # ---------------- Display ----------------
    st.subheader("Point Predictions")
    st.info(f"Factor level (now): **{factor_now:.2f} IU/dL**")
    st.info(f"Bleed Risk: **{risk_label}**  (p = {proba*100:.1f}%, threshold = {best_thr:.2f})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, preds, marker="o")
    ax.axhline(y=50, color="r", linestyle="--", label="Safe Threshold (50 IU/dL)")
    ax.set_xlabel("Time Ahead (hours)")
    ax.set_ylabel("Predicted Factor Level (IU/dL)")
    ax.set_title("Temporal Prediction")
    ax.legend()
    st.pyplot(fig)

    st.caption("Tip: Toggle “Enforce monotonic decay” if no new infusion is expected within the horizon.")