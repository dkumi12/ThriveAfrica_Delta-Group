"""
Diabetes Prediction Web Application
===================================
A Streamlit web app for predicting diabetes risk using machine learning.

Author: Delta Group - Thrive Africa
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessor."""
    try:
        # Try relative paths first
        model_path = Path("../models/diabetes_model.pkl")
        preprocessor_path = Path("../models/preprocessor.pkl")
        
        if not model_path.exists():
            # Fallback to current directory
            model_path = Path("diabetes_model.pkl")
            preprocessor_path = Path("preprocessor.pkl")
            
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def calculate_derived_features(glucose, insulin, skin_thickness, bmi, age):
    """Calculate derived features for prediction."""
    # Avoid division by zero
    glucose_insulin_ratio = glucose / max(insulin, 1e-6)
    glucose_insulin_ratio = min(glucose_insulin_ratio, 1e6)
    
    skin_thickness_bmi_ratio = skin_thickness / max(bmi, 1e-6)
    skin_thickness_bmi_ratio = min(skin_thickness_bmi_ratio, 1e6)
    
    glucose_bmi = np.log1p(min(glucose * bmi, 1e6))
    
    # Age group
    if age < 25:
        age_group = "<25"
    elif age <= 40:
        age_group = "25-40"
    elif age <= 60:
        age_group = "40-60"
    else:
        age_group = ">60"
    
    return glucose_insulin_ratio, skin_thickness_bmi_ratio, glucose_bmi, age_group

def get_risk_interpretation(probability):
    """Interpret the risk probability."""
    if probability < 0.3:
        return "Low Risk", "🟢", "Your risk appears to be low. Continue maintaining a healthy lifestyle!"
    elif probability < 0.6:
        return "Moderate Risk", "🟡", "You have moderate risk. Consider consulting a healthcare provider for preventive measures."
    else:
        return "High Risk", "🔴", "Your risk appears to be high. Please consult a healthcare provider for proper evaluation and guidance."

def main():
    """Main application function."""
    # Load model artifacts
    model, preprocessor = load_model_artifacts()
    
    # App header
    st.markdown('<h1 class="main-header">🏥 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application uses a machine learning model to assess diabetes risk based on various health parameters. 
    Please enter your information below to get a risk assessment.
    
    **⚠️ Important:** This tool is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Sidebar for input
    st.sidebar.header("📋 Enter Your Information")
    
    with st.sidebar:
        st.subheader("Basic Information")
        pregnancies = st.slider("Number of Pregnancies", 0, 20, 0, help="Number of times pregnant")
        age = st.slider("Age (years)", 0, 120, 30, help="Your current age")
        
        st.subheader("Health Measurements")
        glucose = st.slider("Glucose Level (mg/dL)", 0.0, 200.0, 100.0, 
                           help="Plasma glucose concentration (normal: 70-99 mg/dL fasting)")
        blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 150, 70, 
                                  help="Diastolic blood pressure (normal: <80 mmHg)")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, 
                                  help="Triceps skin fold thickness")
        insulin = st.slider("Insulin Level (mu U/ml)", 0.0, 1000.0, 80.0, 
                           help="2-Hour serum insulin (normal: <25 mu U/ml)")
        bmi = st.slider("BMI (kg/m²)", 0.0, 70.0, 25.0, 
                       help="Body Mass Index (normal: 18.5-24.9)")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Your Health Profile")
        
        # Display current values in a nice format
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Glucose", f"{glucose:.1f} mg/dL")
            st.metric("Blood Pressure", f"{blood_pressure} mmHg")
            st.metric("BMI", f"{bmi:.1f} kg/m²")
        
        with metrics_col2:
            st.metric("Age", f"{age} years")
            st.metric("Pregnancies", f"{pregnancies}")
            st.metric("Skin Thickness", f"{skin_thickness} mm")
        
        with metrics_col3:
            st.metric("Insulin", f"{insulin:.1f} mu U/ml")
            
            # BMI category
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
            st.metric("BMI Category", bmi_category)
    
    with col2:
        st.subheader("🔮 Risk Assessment")
        
        if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
            # Calculate derived features
            glucose_insulin_ratio, skin_thickness_bmi_ratio, glucose_bmi, age_group = calculate_derived_features(
                glucose, insulin, skin_thickness, bmi, age
            )
            
            # Create input DataFrame
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
            
            try:
                # Make prediction
                input_scaled = preprocessor.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                # Get risk interpretation
                risk_level, risk_icon, risk_message = get_risk_interpretation(probability)
                
                # Display results
                st.markdown("### 📋 Results")
                
                # Risk level with color coding
                if probability >= 0.6:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3>{risk_icon} {risk_level}</h3>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                        <p>{risk_message}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3>{risk_icon} {risk_level}</h3>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                        <p>{risk_message}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 💡 Risk Factors Analysis")
                
                risk_factors = []
                if glucose > 126:
                    risk_factors.append("High glucose level (>126 mg/dL)")
                if blood_pressure > 80:
                    risk_factors.append("High blood pressure (>80 mmHg)")
                if bmi > 30:
                    risk_factors.append("High BMI (>30 kg/m²)")
                if age > 45:
                    risk_factors.append("Age above 45 years")
                
                if risk_factors:
                    st.warning("⚠️ **Elevated Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ **Good News:** Your key health parameters are within normal ranges!")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p><strong>Developed by Delta Group - Thrive Africa</strong></p>
        <p>⚠️ <em>This tool is for educational purposes only. Always consult with healthcare professionals for medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
