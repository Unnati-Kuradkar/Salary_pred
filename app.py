import os
import streamlit as st
import numpy as np

# Try both loaders
import joblib
import pickle

# --- Setup path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "linear_regression_model.pkl")

# --- Load model safely ---
model = None

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Place 'linear_regression_model.pkl' in the same folder as app.py")
    st.stop()

# Try joblib first
try:
    model = joblib.load(model_path)
except Exception:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.error(f"Details: {e}")
        st.info("👉 Fix: Re-save the model using the SAME Python & sklearn version, or use joblib.")
        st.stop()

# --- UI ---
st.set_page_config(page_title="Salary Predictor", page_icon="💼")

st.title("💼 Salary Prediction App")
st.write("Enter your details below:")

# --- Inputs (adjust if your model uses more features) ---
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

# --- Predict ---
if st.button("Predict Salary"):
    try:
        input_data = np.array([[experience]])
        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Salary: {prediction[0]:,.2f}")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.error(f"Details: {e}")
