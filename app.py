import streamlit as st
import pandas as pd
import pickle

# =========================
# Load Saved Files
# =========================
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# =========================
# App Title
# =========================
st.set_page_config(
    page_title="Hair Fall Prediction",
    page_icon="💇",
    layout="centered"
)

st.title("💇 Hair Fall Severity Prediction System")

st.write(
    "Predict hair fall condition based on daily lifestyle factors."
)

# =========================
# User Inputs
# =========================

age = st.selectbox(
    "Age",
    ['18-20', '21-25', '26-30']
)

gender = st.selectbox(
    "Gender",
    ['Male', 'Female']
)

stress = st.selectbox(
    "Stress Level",
    ['Low', 'Moderate', 'High']
)

sleep = st.selectbox(
    "Sleep Duration",
    ['Less than 5 hours', '5-7 hours', 'More than 7 hours']
)

diet = st.selectbox(
    "Diet Quality",
    ['Poor', 'Average', 'Good']
)

water = st.selectbox(
    "Water Intake",
    ['Low', 'Moderate', 'High']
)

smoking = st.selectbox(
    "Smoking",
    ['Yes', 'No']
)

exercise = st.selectbox(
    "Exercise Frequency",
    ['Never', 'Sometimes', 'Regular']
)

# =========================
# Scalp Condition
# =========================

st.subheader("Scalp Condition")

dandruff = st.checkbox("Dandruff")
oily_scalp = st.checkbox("Oily Scalp")
itching = st.checkbox("Itching")

# =========================
# Prediction Button
# =========================

if st.button("Predict Hair Fall"):

    try:

        # =========================
        # Input Data
        # =========================

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

        # Convert into DataFrame
        input_df = pd.DataFrame([input_data])

        # =========================
        # Encode Categorical Columns
        # =========================

        for col in input_df.columns:

            if col in label_encoders:

                le = label_encoders[col]

                input_df[col] = le.transform(input_df[col])

        # =========================
        # Scale Data
        # =========================

        scaled_data = scaler.transform(input_df)

        # =========================
        # Prediction
        # =========================

        prediction = model.predict(scaled_data)[0]

        # =========================
        # Decode Prediction
        # =========================

        target_encoder = label_encoders['Current Hair Fall Severity']

        result = target_encoder.inverse_transform([prediction])[0]

        # =========================
        # Show Result
        # =========================

        st.success(f"Predicted Hair Fall Condition: {result}")

    except Exception as e:

        st.error(f"Error: {e}")
