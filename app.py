import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessor
model = joblib.load('logistic_regression_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Title and description
st.title("Diabetes Prediction App")
st.write("""
This app uses a logistic regression model to predict diabetes risk based on patient data. 
Enter the patient details below and click 'Predict' to see the result.
""")

# Input fields for raw features
st.header("Patient Data Input")
pregnancies = st.slider("Pregnancies", 0, 20, 0, help="Number of times pregnant")
glucose = st.slider("Glucose (mg/dL)", 0.0, 200.0, 100.0, help="Plasma glucose concentration")
blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 150, 70, help="Diastolic blood pressure")
skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
insulin = st.slider("Insulin (mu U/ml)", 0.0, 1000.0, 80.0, help="2-Hour serum insulin")
bmi = st.slider("BMI (kg/m²)", 0.0, 70.0, 25.0, help="Body mass index")
age = st.slider("Age (years)", 0, 120, 30, help="Age of the patient")

# Derive Age_Group based on age
if age < 25:
    age_group = "<25"
elif age <= 40:
    age_group = "25-40"
elif age <= 60:
    age_group = "40-60"
else:
    age_group = ">60"

# Compute derived features
glucose_insulin_ratio = glucose / (insulin + 1e-6)  # Avoid division by zero
glucose_insulin_ratio = np.clip(glucose_insulin_ratio, a_min=None, a_max=1e6)
skin_thickness_bmi_ratio = skin_thickness / (bmi + 1e-6)  # Avoid division by zero
skin_thickness_bmi_ratio = np.clip(skin_thickness_bmi_ratio, a_min=None, a_max=1e6)
glucose_bmi = glucose * bmi
glucose_bmi = np.log1p(np.clip(glucose_bmi, a_min=None, a_max=1e6))  # Log transform and clip

# Create a DataFrame with all features
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'Age': [age],
    'Glucose_Insulin_Ratio': [glucose_insulin_ratio],
    'SkinThickness_BMI_Ratio': [skin_thickness_bmi_ratio],
    'Glucose_BMI': [glucose_bmi],
    'Age_Group': [age_group]
})

# Prediction button
if st.button("Predict"):
    # Preprocess the input data
    input_scaled = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of diabetes
    
    # Display results with bold text using markdown
    st.subheader("Prediction Result")
    if prediction == 1:
        st.markdown(f"**Prediction**: Diabetes (Positive)")
        st.markdown(f"**Probability of Diabetes**: {probability:.2%}")
    else:
        st.markdown(f"**Prediction**: No Diabetes (Negative)")
        st.markdown(f"**Probability of Diabetes**: {probability:.2%}")
    # Removed the note about precision and recall as requested

# Footer with updated text
st.write("Developed for demonstration by Delta Group_ThriveAfrica.")
# Removed the "Show Feature Importance" checkbox and associated code