import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown

# ==============================
# File base path (works local & cloud)
# ==============================
BASE = os.path.dirname(__file__)

# ==============================
# Load trained artifacts
# ==============================
model = joblib.load(os.path.join(BASE, "best_asteroid_hazard_model.pkl"))
imputer = joblib.load(os.path.join(BASE, "feature_imputer.pkl"))
scaler = joblib.load(os.path.join(BASE, "feature_scaler.pkl"))
meta = joblib.load(os.path.join(BASE, "model_metadata.pkl"))

features = meta["feature_names"]
log_cols = meta["log_transform_cols"]
threshold = float(meta["optimal_threshold"])

# ==============================
# Download + Load NASA dataset
# ==============================
DATA_FILE = os.path.join(BASE, "nasa_neows.csv")
FILE_ID = "1BN6ro6Qtfd4j7tUtlrgVzZ_Ts3axj6v0"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, DATA_FILE, quiet=False)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df[features].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

df = load_data()

# ==============================
# Pick a real asteroid
# ==============================
if "current" not in st.session_state:
    st.session_state.current = df.sample(1).iloc[0]

st.sidebar.header("Asteroid Control")

if st.sidebar.button("Load Random Asteroid"):
    st.session_state.current = df.sample(1).iloc[0]

if st.sidebar.button("Load Random PHO"):
    pho = df[(df["Minimum Orbit Intersection"] <= 0.05) &
             (df["Absolute Magnitude"] <= 22)]
    st.session_state.current = pho.sample(1).iloc[0]

row = st.session_state.current

# ==============================
# User control of physics
# ==============================
H = st.sidebar.number_input("Absolute Magnitude (H)", 10.0, 30.0, float(row["Absolute Magnitude"]))
moid = st.sidebar.number_input("Minimum Orbit Intersection (AU)", 0.0, 1.0, float(row["Minimum Orbit Intersection"]))
ecc = st.sidebar.number_input("Eccentricity", 0.0, 1.5, float(row["Eccentricity"]))

# ==============================
# Build full feature vector
# ==============================
x = row.copy()
x["Absolute Magnitude"] = H
x["Minimum Orbit Intersection"] = moid
x["Eccentricity"] = ecc

x = pd.DataFrame([x], columns=features)
x = x.apply(pd.to_numeric, errors="coerce")

# ==============================
# Log transforms (same as training)
# ==============================
for col in log_cols:
    if x[col].iloc[0] > 0:
        x[col] = np.log(x[col])

# ==============================
# Impute + Scale
# ==============================
x = imputer.transform(x)
x = scaler.transform(x)

# ==============================
# Predict
# ==============================
prob = model.predict_proba(x)[0][1]
pred = prob >= threshold
physics = (moid <= 0.05) and (H <= 22)

# ==============================
# UI
# ==============================
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
