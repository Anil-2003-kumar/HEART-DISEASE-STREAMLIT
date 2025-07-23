import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("💓 Heart Disease Risk Predictor")

# DEBUG: Show files in the current directory
st.write("📂 Files in this folder:", os.listdir())

# Load model
try:
    model = joblib.load("heart_disease_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Inputs
age = st.number_input('Age', 1, 120, 50)
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cp = st.selectbox('Chest Pain Type (0 = typical angina ... 3 = asymptomatic)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120? (1 = Yes, 0 = No)', [0, 1])
restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', 60, 220, 150)
exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1])
oldpeak = st.number_input('ST depression', 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox('Slope of the ST segment', [0, 1, 2])
ca = st.selectbox('Number of major vessels (0–3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible)', [1, 2, 3])

# Predict
if st.button("Predict"):
    try:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]])
        st.write("📊 Input data:", input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High risk of heart disease (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Low risk of heart disease (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
