import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Hair Fall Prediction", layout="wide")

# ============================================
# LOAD MODEL ARTIFACTS - CORRECT WAY
# ============================================
@st.cache_resource
def load_artifacts():
    """Load all model artifacts safely"""
    try:
        # Check if files exist
        required_files = ['model.pkl', 'scaler.pkl', 'label_encoders.pkl', 'feature_names.pkl']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"❌ Missing files: {missing_files}")
            st.info("Please upload all required model files")
            return None, None, None, None
        
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, label_encoders, feature_names
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Load artifacts
model, scaler, label_encoders, feature_names = load_artifacts()

# Check if loading was successful
if model is None:
    st.stop()

# ============================================
# UI TITLE
# ============================================
st.title("🧑‍🦲 Hair Fall Severity Prediction System")
st.markdown("---")
st.success(f"✅ Model loaded successfully with **{len(feature_names)}** features")

# Display feature names in expander
with st.expander("📋 Model Features"):
    st.write(f"Total features: {len(feature_names)}")
    for i, f in enumerate(feature_names, 1):
        st.write(f"{i}. {f}")

st.markdown("---")

# ============================================
# CREATE INPUT FORM
# ============================================
st.subheader("📋 Fill Your Information")

# Create two columns for layout
col1, col2 = st.columns(2)

# Dictionary to store inputs
input_data = {}

# Define common options for features
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
    # Alternate between columns
    current_col = col1 if i % 2 == 0 else col2
    
    # For binary features (scalp conditions)
    if feature in ['Dandruff', 'Oily_Scalp', 'Itching']:
        input_data[feature] = 1 if current_col.checkbox(feature.replace('_', ' ')) else 0
    
    # For features with predefined options
    elif feature in feature_options:
        input_data[feature] = current_col.selectbox(feature, feature_options[feature])
    
    # For features with custom detection
    else:
        # Try to detect by keyword
        feature_lower = feature.lower()
        
        if 'age' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Age'])
        elif 'gender' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Gender'])
        elif 'diet' in feature_lower and 'quality' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Diet Quality'])
        elif 'exercise' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Exercise Frequency'])
        elif 'sleep' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Sleep Duration'])
        elif 'stress' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Stress Level'])
        elif 'smoking' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Smoking Habit'])
        elif 'protein' in feature_lower:
            input_data[feature] = current_col.selectbox(feature, feature_options['Dietary Protein Intake'])
        else:
            input_data[feature] = current_col.text_input(feature, "Normal")

# ============================================
# PREDICTION BUTTON
# ============================================
if st.button("🔮 Predict Hair Fall Severity", type="primary"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        st.write("**Debug - Input Data:**")
        st.json(input_data)
        
        # Encode categorical columns
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    # Get the encoder
                    encoder = label_encoders[col]
                    value = input_df[col].iloc[0]
                    
                    # Check if value exists in encoder classes
                    if value in encoder.classes_:
                        input_df[col] = encoder.transform([value])[0]
                    else:
                        # Use the first class as default
                        st.warning(f"Value '{value}' not seen in training, using default")
                        input_df[col] = 0
                except Exception as e:
                    st.warning(f"Encoding error for {col}: {str(e)}")
                    input_df[col] = 0
        
        # Convert boolean to int
        for col in input_df.columns:
            if input_df[col].dtype == bool:
                input_df[col] = input_df[col].astype(int)
        
        # Ensure all features are in correct order
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        
        # ============================================
        # DISPLAY RESULT
        # ============================================
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Severity mapping
        severity_map = {0: "Low", 1: "Mild", 2: "Moderate", 3: "Severe"}
        severity = severity_map.get(prediction, "Unknown")
        
        # Display based on severity
        if prediction >= 2:
            st.error(f"⚠️ **High Hair Fall Severity: {severity}**")
            st.warning("**Recommendations:**\n- Consult a dermatologist\n- Improve diet with more protein\n- Reduce stress\n- Get adequate sleep (7-8 hours)")
        elif prediction == 1:
            st.warning(f"📈 **Mild Hair Fall Severity: {severity}**")
            st.info("**Recommendations:**\n- Increase protein intake\n- Regular exercise\n- Proper hair care routine")
        else:
            st.success(f"✅ **Low Hair Fall Severity: {severity}**")
            st.info("**Recommendations:**\n- Maintain current healthy habits\n- Continue regular hair care")
        
        # Show confidence scores
        st.write("**Confidence Levels:**")
        for i, prob in enumerate(probabilities):
            severity_level = severity_map.get(i, f"Level {i}")
            st.progress(float(prob), text=f"{severity_level}: {prob:.1%}")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("**Debug Info:**")
        st.write(f"Expected features: {feature_names}")
        st.write(f"Provided features: {list(input_data.keys())}")
        
        # Show the actual input data types
        st.write("**Input data types:**")
        for key, value in input_data.items():
            st.write(f"{key}: {type(value)} = {value}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("💡 **Note:** This prediction is based on machine learning model and should not replace professional medical advice.")
