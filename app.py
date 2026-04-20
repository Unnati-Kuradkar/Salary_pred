import streamlit as st
import pandas as pd
import pickle
import os

# -------------------- LOAD MODEL --------------------
model_path = "linear_regression_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Upload 'linear_regression_model.pkl' to repo.")
    st.stop()

data = pickle.load(open(model_path, "rb"))

# Handle both cases (model only OR dict with model + columns)
if isinstance(data, dict):
    model = data["model"]
    model_columns = data["columns"]
else:
    model = data
    model_columns = None

# -------------------- UI --------------------
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Enter the details below to predict salary.")

# -------------------- INPUTS --------------------
age = st.number_input("Age", 18, 65, 25)
experience = st.number_input("Years of Experience", 0, 40, 1)

gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])

# -------------------- PREPROCESS --------------------
def preprocess_input():
    input_dict = {
        "Age": age,
        "Years of Experience": experience,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Education Level_Master's": 1 if education == "Master's" else 0,
        "Education Level_PhD": 1 if education == "PhD" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # Match training columns (VERY IMPORTANT)
    if model_columns is not None:
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

    return input_df

# -------------------- PREDICTION --------------------
if st.button("Predict Salary"):
    try:
        input_df = preprocess_input()
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Salary: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
