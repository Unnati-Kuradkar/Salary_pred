import os
import pickle
import streamlit as st
import numpy as np

# --- Load model safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "linear_regression_model.pkl")

# Debug (optional - remove later)
# st.write("Model path:", model_path)
# st.write("Exists:", os.path.exists(model_path))

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'linear_regression_model.pkl' is in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- UI ---
st.title("💼 Salary Prediction App")

st.write("Enter details to predict salary")

# Example input (modify based on your model)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

# --- Prediction ---
if st.button("Predict Salary"):
    try:
        input_data = np.array([[experience]])
        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Salary: {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
