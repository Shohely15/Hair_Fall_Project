import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.set_page_config(page_title="Hair Fall Prediction", layout="wide")

# ============================================
# LOAD MODEL ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    try:
        # Check if files exist
        required_files = ['model.pkl', 'scaler.pkl', 'label_encoders.pkl', 'feature_names.pkl']
        missing = [f for f in required_files if not os.path.exists(f)]
        
        if missing:
            st.error(f"Missing files: {missing}")
            st.info("Please make sure all model files are uploaded")
            st.stop()
        
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, scaler, label_encoders, feature_names = load_artifacts()

# ============================================
# UI TITLE
# ============================================
st.title("🧑‍🦲 Hair Fall Severity Prediction System")
st.markdown("---")
st.info(f"✅ Model loaded successfully with **{len(feature_names)}** features")

# ============================================
# CREATE INPUT FORM
# ============================================
st.subheader("📋 Fill Your Information")

input_data = {}

# Split features into two columns
col1, col2 = st.columns(2)

# Define default options for common features
feature_options = {
    'Age': ['18-20', '21-25', '26-30', '31-35', '36-40', '40+'],
    'Gender': ['Male', 'Female', 'Other'],
    'Diet Quality': ['Poor', 'Average', 'Good', 'Excellent'],
    'Exercise Frequency': ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'Sleep Duration': ['<5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '>8 hours'],
    'Stress Level': ['Low', 'Medium', 'High', 'Very High'],
    'Smoking Habit': ['Yes', 'No'],
    'Dietary Protein Intake': ['Low', 'Medium', 'High'],
    'Hair Wash Frequency': ['Daily', 'Alternate days', '2-3 times/week', 'Weekly', 'Rarely'],
    'Water Intake': ['<2L', '2-3L', '3-4L', '>4L']
}

# Create input fields for each feature
for i, feature in enumerate(feature_names):
    # Choose column
    current_col = col1 if i < len(feature_names)//2 else col2
    
    # For binary/scalp features
    if feature in ['Dandruff', 'Oily_Scalp', 'Itching']:
        input_data[feature] = 1 if current_col.checkbox(feature.replace('_', ' ')) else 0
    
    # For features with predefined options
    elif feature in feature_options:
        input_data[feature] = current_col.selectbox(feature, feature_options[feature])
    
    # For numeric features
    elif feature in ['Hair Fall Severity']:
        input_data[feature] = current_col.slider(feature, 0, 3, 1)
    
    # Default text input
    else:
        input_data[feature] = current_col.text_input(feature, "Normal")

# ============================================
# PREDICTION BUTTON
# ============================================
if st.button("🔮 Predict Hair Fall Severity", type="primary"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical columns
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    if input_df[col].iloc[0] in label_encoders[col].classes_:
                        input_df[col] = label_encoders[col].transform(input_df[col])
                    else:
                        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
                except:
                    input_df[col] = 0
        
        # Convert boolean to int
        for col in input_df.columns:
            if input_df[col].dtype == bool:
                input_df[col] = input_df[col].astype(int)
        
        # Ensure correct feature order
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Get probability
        try:
            proba = model.predict_proba(input_scaled)[0]
            confidence = max(proba) * 100
        except:
            confidence = None
        
        # ============================================
        # DISPLAY RESULT
        # ============================================
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        severity_map = {0: "Low", 1: "Mild", 2: "Moderate", 3: "Severe"}
        severity = severity_map.get(prediction, "Unknown")
        
        # Display based on severity
        if prediction >= 2:
            st.error(f"⚠️ **High Hair Fall Severity: {severity}**")
            with st.expander("📝 Recommendations"):
                st.write("""
                - 👨‍⚕️ **Consult a dermatologist immediately**
                - 🥗 **Improve diet with more protein and iron**
                - 😴 **Ensure 7-8 hours of quality sleep**
                - 🧘 **Reduce stress through meditation/exercise**
                - 💆 **Use mild, sulfate-free hair products**
                """)
        elif prediction == 1:
            st.warning(f"📈 **Mild Hair Fall Severity: {severity}**")
            with st.expander("📝 Recommendations"):
                st.write("""
                - 🥚 **Increase protein intake (eggs, fish, nuts)**
                - 🏃 **Regular exercise (30 mins daily)**
                - 💤 **Maintain consistent sleep schedule**
                - 🧴 **Use hair strengthening products**
                """)
        else:
            st.success(f"✅ **Low Hair Fall Severity: {severity}**")
            with st.expander("📝 Recommendations"):
                st.write("""
                - 🌟 **Maintain current healthy habits**
                - 🥬 **Continue balanced diet**
                - 💧 **Stay hydrated**
                - 🧴 **Regular hair care routine**
                """)
        
        # Show confidence
        if confidence:
            st.write(f"**Confidence: {confidence:.1f}%**")
            st.progress(confidence/100)
        
        # Show all inputs
        with st.expander("📋 Your Input Data"):
            st.json(input_data)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Expected features:", feature_names)
        st.write("Provided features:", list(input_data.keys()))
