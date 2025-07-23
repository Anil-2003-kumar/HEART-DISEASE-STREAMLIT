import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('heart_disease_model.pkl')

st.title("💓 Heart Disease Risk Predictor")

# User input fields
age = st.number_input('Age', 1, 120, 50)
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', 80, 200, 120)
chol = st.number_input('Cholesterol', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120? (0 = No, 1 = Yes)', [0, 1])
restecg = st.selectbox('Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', 60, 220, 150)
exang = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
oldpeak = st.number_input('ST Depression', 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox('Slope of ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)', [0, 1, 2])
ca = st.selectbox('Major Vessels Colored (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible)', [1, 2, 3])

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease (Risk: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of heart disease (Risk: {probability:.2f})")

