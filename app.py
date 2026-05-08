import streamlit as st
import pandas as pd
import pickle
import re

# Load model artifacts
with open('model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
feature_names = artifacts['feature_names']
target_classes = artifacts['target_classes']

st.set_page_config(page_title="Hair Fall Prediction", layout="wide")
st.title("🧑‍🦲 Hair Fall Severity Prediction System")
st.markdown("---")

# Input form
st.subheader("📋 Fill Your Information")

col1, col2 = st.columns(2)

with col1:
    # Age selection
    if 'Age' in feature_names:
        age_options = ['18-20', '21-25', '26-30', '31-35', '36-40', '40+']
        age = st.selectbox("Age", age_options)
    
    # Gender selection
    if 'Gender' in feature_names:
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    
    # Diet Quality
    if 'Diet Quality' in feature_names:
        diet_quality = st.selectbox("Diet Quality", ['Poor', 'Average', 'Good', 'Excellent'])
    
    # Exercise Frequency
    if 'Exercise Frequency' in feature_names:
        exercise_freq = st.selectbox("Exercise Frequency", ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])

with col2:
    # Sleep Duration
    if 'Sleep Duration' in feature_names:
        sleep_duration = st.selectbox("Sleep Duration", ['<5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '>8 hours'])
    
    # Stress Level
    if 'Stress Level' in feature_names:
        stress_level = st.selectbox("Stress Level", ['Low', 'Medium', 'High', 'Very High'])
    
    # Hair Wash Frequency
    if 'Hair Wash Frequency' in feature_names:
        hair_wash = st.selectbox("Hair Wash Frequency", ['Daily', 'Alternate days', '2-3 times/week', 'Weekly', 'Rarely'])
    
    # Smoking Habit
    if 'Smoking Habit' in feature_names:
        smoking = st.selectbox("Smoking Habit", ['Yes', 'No'])

# Additional features (Scalp related)
st.subheader("🧴 Scalp Conditions")
scalp_col1, scalp_col2, scalp_col3 = st.columns(3)

with scalp_col1:
    dandruff = st.checkbox("Dandruff")
with scalp_col2:
    oily_scalp = st.checkbox("Oily Scalp")
with scalp_col3:
    itching = st.checkbox("Itching")

# Nutrition
st.subheader("🥗 Nutrition Information")
nut_col1, nut_col2 = st.columns(2)

with nut_col1:
    if 'Dietary Protein Intake' in feature_names:
        protein_intake = st.selectbox("Dietary Protein Intake", ['Low', 'Medium', 'High'])
with nut_col2:
    if 'Water Intake' in feature_names:
        water_intake = st.selectbox("Water Intake", ['<2L', '2-3L', '3-4L', '>4L'])

# Prediction button
if st.button("🔮 Predict Hair Fall Severity", type="primary"):
    # Create input dictionary
    input_dict = {}
    
    # Add all features
    for feature in feature_names:
        if feature == 'Age':
            input_dict[feature] = age
        elif feature == 'Gender':
            input_dict[feature] = gender
        elif feature == 'Diet Quality':
            input_dict[feature] = diet_quality
        elif feature == 'Exercise Frequency':
            input_dict[feature] = exercise_freq
        elif feature == 'Sleep Duration':
            input_dict[feature] = sleep_duration
        elif feature == 'Stress Level':
            input_dict[feature] = stress_level
        elif feature == 'Smoking Habit':
            input_dict[feature] = smoking
        elif feature == 'Dietary Protein Intake':
            input_dict[feature] = protein_intake
        elif feature == 'Water Intake':
            input_dict[feature] = water_intake
        elif feature == 'Hair Wash Frequency':
            input_dict[feature] = hair_wash
        elif feature == 'Dandruff':
            input_dict[feature] = dandruff
        elif feature == 'Oily_Scalp':
            input_dict[feature] = oily_scalp
        elif feature == 'Itching':
            input_dict[feature] = itching
    
    # Create DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical variables
    for col in input_df.columns:
        if col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except:
                # Handle unseen labels
                input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]
    
    # Scale features
    input_scaled = scaler.transform(input_df[feature_names])
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display result
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    severity_levels = {0: "Low", 1: "Mild", 2: "Moderate", 3: "Severe"}
    severity = severity_levels.get(prediction, str(prediction))
    
    if prediction >= 2:
        st.error(f"⚠️ **High Hair Fall Severity: {severity}**")
    elif prediction == 1:
        st.warning(f"📈 **Moderate Hair Fall Severity: {severity}**")
    else:
        st.success(f"✅ **Low Hair Fall Severity: {severity}**")
    
    # Show confidence
    st.write("**Confidence Score:**")
    for i, prob in enumerate(prediction_proba):
        st.write(f"- {severity_levels.get(i, str(i))}: {prob:.1%}")
