# Import libraries and dependencies
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Import ML model
model = joblib.load("lr_stroke_model.pkl")

#import Scaler
scaler = joblib.load('scaler.pkl')

# Create App Headline
st.title("Are You At Risk of Having Stroke? :health_worker:")

# Create health information input options
gender = st.selectbox("What is Your Gender?", ["Male", "Female", "Other"])
age = st.select_slider("What is Your Age?", range(1,101))
hypertension = st.radio("Hypertension?", ["Yes", 'No'])
heart_disease = st.radio("Heart Disease?", ["Yes", "No"])
ever_married = st.radio("Ever Married?", ["Yes", "No"])
work_type = st.selectbox("Employment Type?", ["Private Sector", "Public Sector", "Self Employed", "Parent", "Never Worked"])
Residence_type = st.selectbox("Residence Type?", ["Urban", "Rural"])
avg_glucose_level = st.select_slider("Average Glucose Level", range(50,301))
bmi = st.select_slider("BMI", range(10,101))
smoking_status = st.selectbox("Smoking Status", ["Currently Smokes", 'Previously Smoked', "Never Smoked", "Not Sure"])

# Compute one-hot encoding for number_of_risks
def compute_risks(input_data):
    risks = {
        'number_of_risks_0': 0,
        'number_of_risks_1': 0,
        'number_of_risks_2': 0,
        'number_of_risks_3': 0,
        'number_of_risks_4': 0,
        'number_of_risks_5': 0
    }

    counter = 0
    if input_data['avg_glucose_level'] >= 140:
        counter += 1
    if input_data['heart_disease'] == 1:
        counter += 1
    if input_data['hypertension'] == 1:
        counter += 1
    if (input_data['smoking_status_smokes'] == 1 or 
        input_data['smoking_status_formerly smoked'] == 1):
        counter += 1
    if input_data['bmi'] > 25:
        counter += 1
        
    risks[f'number_of_risks_{counter}'] = 1
    return risks

input_data = {
    'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender_Female': 1 if gender == "Female" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 1 if gender == "Other" else 0,
    'ever_married_No': 1 if ever_married == "No" else 0,
    'ever_married_Yes': 1 if ever_married == "Yes" else 0,
    'work_type_Govt_job': 1 if work_type == "Public Sector" else 0,
    'work_type_Never_worked': 1 if work_type == "Never Worked" else 0,
    'work_type_Private': 1 if work_type == "Private Sector" else 0,
    'work_type_Self-employed': 1 if work_type == "Self Employed" else 0,
    'work_type_children': 1 if work_type == "Parent" else 0,
    'Residence_type_Rural': 1 if Residence_type == "Rural" else 0,
    'Residence_type_Urban': 1 if Residence_type == "Urban" else 0,
    'smoking_status_Unknown': 1 if smoking_status == "Not Sure" else 0,
    'smoking_status_formerly smoked': 1 if smoking_status == "Previously Smoked" else 0,
    'smoking_status_never smoked': 1 if smoking_status == "Never Smoked" else 0,
    'smoking_status_smokes': 1 if smoking_status == "Currently Smokes" else 0
}

# Merge input_data with the results of compute_risks
merged_data = {**input_data, **compute_risks(input_data)}

# Convert to DataFrame
input_df = pd.DataFrame([merged_data])

# Model Prediction
if st.button("Assess Your Risk"):
    # Scale the input data
    scaled_input_df = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input_df)
    
    # Output the result
    if prediction[0] == 1:
        st.write("Based on your inputs, you are at risk of having a stroke. Please consult a medical professional.")
    else:
        st.write("Based on your inputs, you are not currently at risk of having a stroke. Maintain a healthy lifestyle in order to avoid the risk in the future.")