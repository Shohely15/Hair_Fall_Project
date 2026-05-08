import pandas as pd
import pickle
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Load dataset
DF = pd.read_csv("Predicting Hair Loss Severity in University Students with Integrated Lifestyle Analysis in Bangladesh. (Responses) - Form Responses 1.csv")

# -------------------------------
# Cleaning Function
# -------------------------------
def clean_string(s):
    if pd.isna(s):
        return "Normal"
    s = re.sub(r'\(.*?\)', '', str(s)).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Clean categorical columns
for col in DF.select_dtypes(include=['object']).columns:
    DF[col] = DF[col].apply(clean_string)

# Fix Age column formatting
if 'Age' in DF.columns:
    DF['Age'] = DF['Age'].replace({'18–20': '18-20', '18—20': '18-20'})

# -------------------------------
# Feature Engineering - Scalp Condition
# -------------------------------
if 'Scalp Condition' in DF.columns:
    DF['Dandruff'] = DF['Scalp Condition'].apply(lambda x: 1 if 'Dandruff' in str(x) else 0)
    DF['Oily_Scalp'] = DF['Scalp Condition'].apply(lambda x: 1 if 'Oily' in str(x) else 0)
    DF['Itching'] = DF['Scalp Condition'].apply(lambda x: 1 if 'Itching' in str(x) else 0)
    DF.drop(columns=['Scalp Condition'], inplace=True)

# -------------------------------
# Remove unnecessary columns
# -------------------------------
columns_to_remove = ['Timestamp', 'SID', 'Submission ID', 'Email Address', 'Name']
for col in columns_to_remove:
    if col in DF.columns:
        DF.drop(columns=[col], inplace=True)

# -------------------------------
# Handle missing values
# -------------------------------
DF = DF.dropna()

# -------------------------------
# Encode categorical columns
# -------------------------------
label_encoders = {}
categorical_columns = DF.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    DF[col] = le.fit_transform(DF[col])
    label_encoders[col] = le

# -------------------------------
# Prepare features and target
# -------------------------------
TARGET = 'Current Hair Fall Severity'

if TARGET not in DF.columns:
    print(f"Warning: '{TARGET}' column not found!")
    print("Available columns:", DF.columns.tolist())
    exit()

X = DF.drop(TARGET, axis=1)
y = DF[TARGET]

feature_names = X.columns.tolist()
print(f"Features: {feature_names}")
print(f"Target classes: {y.unique()}")

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# -------------------------------
# Handle class imbalance
# -------------------------------
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# -------------------------------
# Scale features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Train Random Forest model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_resampled)

# -------------------------------
# Save model and preprocessing objects
# -------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\n✅ Model files saved successfully!")
print("Files created: model.pkl, scaler.pkl, label_encoders.pkl, feature_names.pkl")
