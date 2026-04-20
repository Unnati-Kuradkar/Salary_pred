
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Set page config for a wider layout
st.set_page_config(layout="wide")

# Load the trained model
try:
    # Ensure 'linear_regression_model.pkl' is accessible.
    # In Colab, files are usually in /content/
    model_path = 'linear_regression_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it's in the correct directory.")
        st.stop()
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Salary Prediction App")
st.write("---")
st.markdown("### Enter the employee features to predict their salary.")

# --- Recreate preprocessing steps for user input ---
# 1. Load original dataset to get categorical values for encoding
try:
    dataset_path = '/content/Salary_Dataset_DataScienceLovers.csv'
    if not os.path.exists(dataset_path):
        st.error(f"Original dataset '{dataset_path}' not found. Cannot set up categorical encoders.")
        st.stop()
    original_df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Error loading original dataset: {e}")
    st.stop()

# 2. Fill NaNs as done during training for consistency (important for mode/mean calculation)
df_for_encoders = original_df.copy() # Use a copy to not modify original_df in this context
for col in df_for_encoders.columns:
    if df_for_encoders[col].dtype == 'object':
        df_for_encoders[col] = df_for_encoders[col].fillna(df_for_encoders[col].mode()[0])
    else:
        df_for_encoders[col] = df_for_encoders[col].fillna(df_for_encoders[col].mean())

# 3. Create and fit LabelEncoders for each categorical column
# These are the columns that were label encoded in the notebook
categorical_cols_to_encode = [
    'Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles'
]
label_encoders = {}
for col in categorical_cols_to_encode:
    if col in df_for_encoders.columns:
        le = LabelEncoder()
        le.fit(df_for_encoders[col].astype(str)) # Fit on the cleaned (NaN filled) categorical column
        label_encoders[col] = le
    else:
        st.warning(f"Column '{col}' not found in dataset for encoding.")

# Input fields for features, mirroring the training features:
# 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'

col1, col2, col3 = st.columns(3)

with col1:
    rating = st.slider("Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    salaries_reported = st.number_input("Salaries Reported", min_value=1, value=1, help="Number of salaries reported for this job role.")
    employment_status_options = list(label_encoders['Employment Status'].classes_)
    employment_status_selected = st.selectbox("Employment Status", options=employment_status_options)
    employment_status_encoded = label_encoders['Employment Status'].transform([employment_status_selected])[0]

with col2:
    company_name_options = list(label_encoders['Company Name'].classes_)
    company_name_selected = st.selectbox("Company Name", options=company_name_options)
    company_name_encoded = label_encoders['Company Name'].transform([company_name_selected])[0]
    location_options = list(label_encoders['Location'].classes_)
    location_selected = st.selectbox("Location", options=location_options)
    location_encoded = label_encoders['Location'].transform([location_selected])[0]

with col3:
    job_title_options = list(label_encoders['Job Title'].classes_)
    job_title_selected = st.selectbox("Job Title", options=job_title_options)
    job_title_encoded = label_encoders['Job Title'].transform([job_title_selected])[0]
    job_roles_options = list(label_encoders['Job Roles'].classes_)
    job_roles_selected = st.selectbox("Job Roles", options=job_roles_options)
    job_roles_encoded = label_encoders['Job Roles'].transform([job_roles_selected])[0]

st.write("---")

if st.button("Predict Salary", help="Click to get the predicted salary based on the entered features."):
    # Create a DataFrame for the input, with encoded categorical values
    input_data = pd.DataFrame([[rating, company_name_encoded, job_title_encoded, salaries_reported, location_encoded, employment_status_encoded, job_roles_encoded]],
                              columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"### Predicted Salary: ₹{prediction:,.2f}")
    st.balloons() # Visual feedback

st.markdown("---T")
st.info("Note: The model used is Linear Regression, which had an R-squared of 0.05. Predictions may not be highly accurate.")

