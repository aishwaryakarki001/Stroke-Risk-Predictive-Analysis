# Import libraries and dependencies
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Import ML model
model = joblib.load("random_forest_classifier_model.joblib")
my_scaler = joblib.load('scaler.save')

# Create App Headline
st.title("Are You At Risk of Having Stroke? ")

# Create health information input options
gender = st.selectbox("What is Your Gender?", ["Male", "Female", "Other"])
age = st.select_slider("What is Your Age?", range(1,101))
hypertension = st.radio("Hypertension?", ["Yes", 'No'])
heart_disease = st.radio("Heart Disease?", ["Yes", "No"])
ever_married = st.radio("Ever Married?", ["Yes", "No"])
work_type = st.selectbox("Employment Type?", ["0", "1", "2", "3", "4"])
Residence_type = st.selectbox("Residence Type?", ["0", "1"])
avg_glucose_level = st.select_slider("Average Glucose Level", range(50,301))
bmi = st.select_slider("BMI", range(10,101))
smoking_status = st.selectbox("Smoking Status", ["0", '1', "2", "3"])


#data[numerical[:-1]] = StandardScaler().fit_transform(data[numerical[:-1]])

input_data = {
    'gender': 1 if gender == "Male" else 0,  
    'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'ever_married': 1 if ever_married == "Yes" else 0,
    'work_type':  work_type ,  # Include work_type directly
    'Residence_type': Residence_type,  # Include Residence_type directly
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status  # Include smoking_status directly
}


# Convert to DataFrame
input_df = pd.DataFrame([input_data])
numerical = input_df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']].columns.tolist()
df_scaled = input_df(my_scaler.fit_transform(input_df), columns = numerical)
#input_df[numerical] = my_scaler.transform(input_df[numerical])

# Model Prediction
if st.button("Assess Your Risk"):

    # Predict
    prediction = model.predict(df_scaled)
    
    # Output the result
    if prediction[0] == 1:
        st.write("Based on your inputs, you are at risk of having a stroke. Please consult a medical professional.")
    else:
        st.write("Based on your inputs, you are not currently at risk of having a stroke. Maintain a healthy lifestyle in order to avoid the risk in the future.")