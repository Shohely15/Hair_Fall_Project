import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load Saved Files
# -------------------------
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# -------------------------
# App Title
# -------------------------
st.title("Hair Fall Severity Prediction System")

st.write("Predict hair fall condition based on lifestyle factors")

# -------------------------
# User Inputs
# -------------------------
age = st.selectbox("Age", ['18-20', '21-25', '26-30'])

gender = st.selectbox("Gender", ['Male', 'Female'])

stress = st.selectbox("Stress Level", ['Low', 'Moderate', 'High'])

sleep = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-7 hours', 'More than 7 hours'])

diet = st.selectbox("Diet Quality", ['Poor', 'Average', 'Good'])

water = st.selectbox("Water Intake", ['Low', 'Moderate', 'High'])

smoking = st.selectbox("Smoking", ['Yes', 'No'])

exercise = st.selectbox("Exercise Frequency", ['Never', 'Sometimes', 'Regular'])

# Scalp Conditions
st.subheader("Scalp Condition")

dandruff = st.checkbox("Dandruff")
oily_scalp = st.checkbox("Oily Scalp")
itching = st.checkbox("Itching")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Hair Fall"):

    input_data = {
        'Age': age,
        'Gender': gender,
        'Stress Level': stress,
        'Sleep Duration': sleep,
        'Diet Quality': diet,
        'Water Intake': water,
        'Smoking': smoking,
        'Exercise Frequency': exercise,
        'Dandruff': int(dandruff),
        'Oily_Scalp': int(oily_scalp),
        'Itching': int(itching)
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    st.success(f"Predicted Hair Fall Condition: {result}")
