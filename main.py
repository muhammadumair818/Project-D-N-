# main.py

import streamlit as st
import numpy as np
import joblib

# =============================
# Load Model and Scaler
# =============================
model = joblib.load("Model/xgb_diabetes_model.pkl")
scaler = joblib.load("Model/scaler.pkl")

# =============================
# Streamlit App UI
# =============================
st.set_page_config(page_title="Diabetes Prediction", page_icon="💉", layout="centered")
st.title("Diabetes Prediction App")
st.write("Enter patient details below to predict whether they are Diabetic or Non-Diabetic.")

# =============================
# Input Fields
# =============================
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bgl = st.number_input("Blood Glucose Level (BGL)", min_value=50.0, max_value=500.0, value=95.0)
diastolic = st.number_input("Diastolic Blood Pressure", min_value=40.0, max_value=200.0, value=80.0)
systolic = st.number_input("Systolic Blood Pressure", min_value=60.0, max_value=250.0, value=120.0)
heart_rate = st.number_input("Heart Rate", min_value=40.0, max_value=200.0, value=90.0)
body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6)
spo2 = st.number_input("SPO2 (%)", min_value=70.0, max_value=100.0, value=97.0)

# =============================
# Predict Button
# =============================
if st.button("Predict"):
    # Convert input to array
    input_data = np.array([[age, bgl, diastolic, systolic, heart_rate, body_temp, spo2]])
    help(st.button)
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]
    
    # Threshold for classificatison
    threshold = 0.35
    prediction = "Diabetic" if prob > threshold else "Non-Diabetic"
    
    # Show results
    st.success(f"Prediction: {prediction}")
    st.info(f"Diabetes Probability: {prob:.4f}")
