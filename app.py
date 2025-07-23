import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('heart_disease_model.pkl')

st.title("Heart Disease Prediction")

age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)', [0, 1])
restecg = st.selectbox('Resting ECG Results (0,1,2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])

if st.button('Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"High risk of heart disease! (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"Low risk of heart disease. (Probability: {prediction_prob:.2f})")

