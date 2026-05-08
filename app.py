import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="Hair Fall Prediction", layout="wide")

# Cache loading to avoid reloading every time
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Try to load feature names if exists
        try:
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except:
            # If feature_names.pkl doesn't exist, get from model
            feature_names = None
        
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Load all artifacts
model, scaler, label_encoders, feature_names = load_artifacts()

st.title("🧑‍🦲 Hair Fall Severity Prediction System")
st.markdown("---")

# Get feature names from model if not available
if feature_names is None:
    st.info("Detecting features from model...")
    feature_names = ['Age', 'Gender', 'Diet Quality', 'Exercise Frequency', 
                     'Sleep Duration', 'Stress Level', 'Smoking Habit', 
                     'Dietary Protein Intake', 'Dandruff', 'Oily_Scalp', 'Itching']

st.subheader("📋 Fill Your Information")

# Create input form
col1, col2 = st.columns(2)

# Dictionary to store inputs
input_data = {}

with col1:
    if 'Age' in feature_names:
        age = st.selectbox("Age", ['18-20', '21-25', '26-30', '31-35', '36-40', '40+'])
        input_data['Age'] = age
    
    if 'Gender' in feature_names:
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        input_data['Gender'] = gender
    
    if 'Diet Quality' in feature_names:
        diet = st.selectbox("Diet Quality", ['Poor', 'Average', 'Good', 'Excellent'])
        input_data['Diet Quality'] = diet
    
    if 'Exercise Frequency' in feature_names:
        exercise = st.selectbox("Exercise Frequency", ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
        input_data['Exercise Frequency'] = exercise

with col2:
    if 'Sleep Duration' in feature_names:
        sleep = st.selectbox("Sleep Duration", ['<5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '>8 hours'])
        input_data['Sleep Duration'] = sleep
    
    if 'Stress Level' in feature_names:
        stress = st.selectbox("Stress Level", ['Low', 'Medium', 'High', 'Very High'])
        input_data['Stress Level'] = stress
    
    if 'Smoking Habit' in feature_names:
        smoking = st.selectbox("Smoking Habit", ['Yes', 'No'])
        input_data['Smoking Habit'] = smoking
    
    if 'Dietary Protein Intake' in feature_names:
        protein = st.selectbox("Dietary Protein Intake", ['Low', 'Medium', 'High'])
        input_data['Dietary Protein Intake'] = protein

# Scalp conditions
st.subheader("🧴 Scalp Conditions (Select all that apply)")
col3, col4, col5 = st.columns(3)

with col3:
    if 'Dandruff' in feature_names:
        dandruff = st.checkbox("Dandruff")
        input_data['Dandruff'] = dandruff

with col4:
    if 'Oily_Scalp' in feature_names:
        oily = st.checkbox("Oily Scalp")
        input_data['Oily_Scalp'] = oily

with col5:
    if 'Itching' in feature_names:
        itching = st.checkbox("Itching")
        input_data['Itching'] = itching

# Additional features if exists
if 'Hair Wash Frequency' in feature_names:
    hair_wash = st.selectbox("Hair Wash Frequency", ['Daily', 'Alternate days', '2-3 times/week', 'Weekly', 'Rarely'])
    input_data['Hair Wash Frequency'] = hair_wash

if 'Water Intake' in feature_names:
    water = st.selectbox("Water Intake", ['<2L', '2-3L', '3-4L', '>4L'])
    input_data['Water Intake'] = water

# Prediction button
if st.button("🔮 Predict Hair Fall Severity", type="primary"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except:
                    # For unseen labels, use the most common or first class
                    input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        
        # Ensure all features from training are present
        if feature_names:
            # Add missing columns with default value 0
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Keep only the features that were used in training
            input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Define severity levels based on your model's output
        severity_map = {0: "Low", 1: "Mild", 2: "Moderate", 3: "Severe"}
        severity = severity_map.get(prediction, "Unknown")
        
        # Show result with appropriate color
        if prediction >= 2:  # Moderate or Severe
            st.error(f"⚠️ **High Hair Fall Severity: {severity}**")
            st.warning("**Recommendations:**\n- Consult a dermatologist\n- Maintain proper diet\n- Reduce stress\n- Use mild hair products")
        elif prediction == 1:  # Mild
            st.warning(f"📈 **Mild Hair Fall Severity: {severity}**")
            st.info("**Recommendations:**\n- Improve diet quality\n- Regular exercise\n- Adequate sleep\n- Gentle hair care routine")
        else:  # Low
            st.success(f"✅ **Low Hair Fall Severity: {severity}**")
            st.info("**Recommendations:**\n- Maintain current healthy habits\n- Continue regular hair care\n- Stay hydrated")
        
        # Show confidence/probability
        try:
            proba = model.predict_proba(input_scaled)[0]
            st.write("**Confidence Levels:**")
            for i, prob in enumerate(proba):
                severity_label = severity_map.get(i, f"Level {i}")
                st.progress(float(prob), text=f"{severity_label}: {prob:.1%}")
        except:
            pass
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please make sure all inputs are filled correctly and try again.")
