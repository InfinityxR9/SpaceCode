import streamlit as st
import numpy as np
import pandas as pd
import joblib

import os
import streamlit as st

# -----------------------------
# Load trained artifacts
# -----------------------------
model = joblib.load("ps3/best_asteroid_hazard_model.pkl")
imputer = joblib.load("ps3/feature_imputer.pkl")
scaler = joblib.load("ps3/feature_scaler.pkl")
meta = joblib.load("ps3/model_metadata.pkl")

features = meta["feature_names"]
log_cols = meta["log_transform_cols"]
threshold = float(meta["optimal_threshold"])

# -----------------------------
# Load NASA data ONCE
# -----------------------------
import gdown
import os

@st.cache_data
def load_data():
    file_id = "1BN6ro6Qtfd4j7tUtlrgVzZ_Ts3axj6v0"
    output = "nasa_neows.csv"

    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

    return pd.read_csv(output, low_memory=False)

df = load_data()

# -----------------------------
# Pick a real asteroid
# -----------------------------
if "current" not in st.session_state:
    st.session_state.current = df.sample(1).iloc[0]

st.sidebar.header("Asteroid Control")

if st.sidebar.button("Load Random Asteroid"):
    st.session_state.current = df.sample(1).iloc[0]

if st.sidebar.button("Load Random PHO"):
    pho = df[(df["Minimum Orbit Intersection"] <= 0.05) & (df["Absolute Magnitude"] <= 22)]
    st.session_state.current = pho.sample(1).iloc[0]

row = st.session_state.current

# -----------------------------
# Allow user to perturb key physics
# -----------------------------
H = st.sidebar.number_input("Absolute Magnitude (H)", 10.0, 30.0, float(row["Absolute Magnitude"]))
moid = st.sidebar.number_input("Minimum Orbit Intersection (AU)", 0.0, 1.0, float(row["Minimum Orbit Intersection"]))
ecc = st.sidebar.number_input("Eccentricity", 0.0, 1.5, float(row["Eccentricity"]))

# -----------------------------
# Build full feature vector
# -----------------------------
x = row[features].copy()
x["Absolute Magnitude"] = H
x["Minimum Orbit Intersection"] = moid
x["Eccentricity"] = ecc
x = np.array(x.values).reshape(1, -1)

# -----------------------------
# Apply training transforms
# -----------------------------
for col in log_cols:
    idx = features.index(col)
    if x[0, idx] > 0:
        x[0, idx] = np.log(x[0, idx])

x = imputer.transform(x)
x = scaler.transform(x)

# -----------------------------
# Predict
# -----------------------------
prob = model.predict_proba(x)[0][1]
pred = prob >= threshold
physics = (moid <= 0.05) and (H <= 22)

# -----------------------------
# Display
# -----------------------------
st.title("🛰 Sentinel Shield")
st.subheader("Near-Earth Object Hazard Classification")

st.metric("Hazard Probability", f"{prob:.4f}")

if pred:
    st.error("⚠ Potentially Hazardous Object")
else:
    st.success("✅ Not Hazardous")

st.write("Decision Threshold:", threshold)
st.write("NASA Physics Rule:", physics)

st.subheader("Model Statistics")
st.write("Model:", meta["model_name"])
st.write("ROC-AUC:", meta["roc_auc"])
st.write("Recall:", meta["recall"])
st.write("Precision:", meta["precision"])
