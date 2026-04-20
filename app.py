import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Enter the details below to predict salary.")

# --- INPUT FIELDS ---
age = st.number_input("Age", min_value=18, max_value=65, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.text_input("Job Title (e.g., Data Scientist)")
experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=1)

# --- PREPROCESS INPUT ---
def preprocess_input(age, gender, education, job_title, experience):
    input_dict = {
        'Age': age,
        'Years of Experience': experience,
    }

    # Example encoding (must match training columns!)
    input_dict['Gender_Male'] = 1 if gender == "Male" else 0
    input_dict["Education Level_Master's"] = 1 if education == "Master's" else 0
    input_dict["Education Level_PhD"] = 1 if education == "PhD" else 0

    # NOTE: Job title encoding depends on your dataset columns
    # Add dummy columns if needed

    return pd.DataFrame([input_dict])

# --- PREDICTION ---
if st.button("Predict Salary"):
    try:
        input_df = preprocess_input(age, gender, education, job_title, experience)

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Salary: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
