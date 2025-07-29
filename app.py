import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Enter your health information to assess your heart disease risk.")

# Input form
with st.form("health_form"):
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    resting_bp = st.slider("Resting BP (mmHg)", 80, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    rest_ecg = st.selectbox("Resting ECG", [0, 1, 2])
    max_hr = st.slider("Max Heart Rate", 60, 210, 140)
    exercise_angina = st.radio("Exercise-induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
    submit = st.form_submit_button("Predict Risk")

if submit:
        input_data = pd.DataFrame([{
        'Age': age,
        'Sex': 1 if sex == "Male" else 0,
        'Chest pain type': chest_pain_type,
        'BP': resting_bp,
        'Cholesterol': cholesterol,
        'FBS over 120': fasting_bs,
        'EKG results': rest_ecg,
        'Max HR': max_hr,
        'Exercise angina': exercise_angina,
        'ST depression': oldpeak,
        'Slope of ST': slope,
        'Number of vessels fluro': ca,
        'Thallium': thal
    }])

    # Predict
    prediction_proba = model.predict_proba(input_data)[0][1]  # class 1 = risk
    risk_percent = prediction_proba * 100

    # Output
    st.subheader(f"🧠 Risk Probability: {risk_percent:.2f}%")

if prediction_proba >= 0.7:
    st.error("🔴 High Risk: Immediate consultation recommended!")
elif prediction_proba >= 0.5:
    st.warning("🟠 Moderate Risk: Regular checkups advised.")
else:
    st.success("🟢 Low Risk: Keep maintaining a healthy lifestyle!")

st.caption("📌 Disclaimer: This is a machine learning based prediction. Always consult a doctor for medical advice.")
