import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('data/salary_dataset.csv')

# Initialize encoders
company_encoder = LabelEncoder()
job_title_encoder = LabelEncoder()
location_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
role_encoder = LabelEncoder()

# Encode categorical columns
df['Company Name'] = company_encoder.fit_transform(df['Company Name'])
df['Job Title'] = job_title_encoder.fit_transform(df['Job Title'])
df['Location'] = location_encoder.fit_transform(df['Location'])
df['Employment Status'] = employment_encoder.fit_transform(df['Employment Status'])
df['Job Roles'] = role_encoder.fit_transform(df['Job Roles'])

# Features & target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('linear_regression_model.pkl', 'wb'))

# Save encoders
pickle.dump(company_encoder, open('encoders/company_encoder.pkl', 'wb'))
pickle.dump(job_title_encoder, open('encoders/job_title_encoder.pkl', 'wb'))
pickle.dump(location_encoder, open('encoders/location_encoder.pkl', 'wb'))
pickle.dump(employment_encoder, open('encoders/employment_encoder.pkl', 'wb'))
pickle.dump(role_encoder, open('encoders/role_encoder.pkl', 'wb'))

print("✅ Model and encoders saved successfully!")
