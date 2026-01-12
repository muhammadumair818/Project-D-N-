
"""
Diabetes Prediction Web Application
====================================
This Streamlit application provides a user-friendly interface for predicting
whether a patient is diabetic or non-diabetic based on vital health parameters.

The model uses XGBoost classification trained on patient health metrics.
"""

import streamlit as st
import numpy as np
import joblib
import os

# Configure page
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load pre-trained model and scaler
try:
    model = joblib.load("Model/xgb_diabetes_model.pkl")
    scaler = joblib.load("Model/scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# ============================================================================
# UI CONFIGURATION & HEADER
# ============================================================================

st.title("🩺 Diabetes Prediction System")
st.markdown("""
This application uses machine learning to predict the likelihood of diabetes
based on important health metrics. Please enter your vital signs below for
an instant prediction.
""")

st.divider()

# ============================================================================
# INPUT SECTION - Patient Health Metrics
# ============================================================================

st.subheader("📋 Patient Health Information")
st.markdown("*Please enter your current vital signs:*")

# Organize inputs in two columns for better UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        help="Patient's age in years"
    )
    diastolic = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=40.0,
        max_value=200.0,
        value=80.0,
        help="Lower blood pressure reading"
    )
    heart_rate = st.number_input(
        "Heart Rate (bpm)",
        min_value=40.0,
        max_value=200.0,
        value=90.0,
        help="Beats per minute"
    )
    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=70.0,
        max_value=100.0,
        value=97.0,
        help="Blood oxygen saturation percentage"
    )

with col2:
    bgl = st.number_input(
        "Blood Glucose (mg/dL)",
        min_value=50.0,
        max_value=500.0,
        value=95.0,
        help="Fasting blood glucose level"
    )
    systolic = st.number_input(
        "Systolic BP (mmHg)",
        min_value=60.0,
        max_value=250.0,
        value=120.0,
        help="Upper blood pressure reading"
    )
    body_temp = st.number_input(
        "Body Temperature (°F)",
        min_value=95.0,
        max_value=105.0,
        value=98.6,
        help="Core body temperature"
    )

st.divider()

# ============================================================================
# PREDICTION SECTION
# ============================================================================

# Prediction button centered
col_button = st.columns([1, 2, 1])
with col_button[1]:
    predict_button = st.button(
        "🔍 Generate Prediction",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    # Prepare input data in correct feature order
    input_data = np.array([[
        age,
        bgl,
        diastolic,
        systolic,
        heart_rate,
        body_temp,
        spo2
    ]])
    
    try:
        # Scale the input using pre-fitted scaler
        input_scaled = scaler.transform(input_data)
        
        # Get prediction probability
        prediction_probability = model.predict_proba(input_scaled)[0][1]
        
        # Classification threshold (tuned for balance between sensitivity & specificity)
        threshold = 0.35
        prediction = "🔴 Diabetic" if prediction_probability > threshold else "🟢 Non-Diabetic"
        
        # Display results with professional styling
        st.divider()
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction_probability > threshold:
                st.error(f"**Prediction: {prediction}**")
            else:
                st.success(f"**Prediction: {prediction}**")
        
        with result_col2:
            st.metric(
                "Confidence Score",
                f"{prediction_probability:.1%}",
                delta=f"{(prediction_probability - 0.5) * 100:.1f}% from neutral"
            )
        
        # Additional context
        st.info("""
        ⚠️ **Disclaimer**: This prediction is for informational purposes only.
        Please consult with a healthcare professional for diagnosis and treatment.
        """)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Footer
st.divider()
st.markdown("""
---
**Model**: XGBoost Classifier | **Threshold**: 0.35  
*For production use, consult with medical professionals.*
""")
