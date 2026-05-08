import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Load dataset
DF = pd.read_csv("Predicting Hair Loss Severity in University Students with Integrated Lifestyle Analysis in Bangladesh.  (Responses) - Form Responses 1.csv")

# -------------------------------
# Cleaning Function
# -------------------------------
def clean_string(s):
    if pd.isna(s):
        return "Normal"

    s = re.sub(r'\(.*?\)', '', str(s)).strip()
    return s

# Clean categorical columns
for col in DF.select_dtypes(include=['object']).columns:
    DF[col] = DF[col].apply(clean_string)

# Example replacements
DF['Age'] = DF['Age'].replace({'18–20': '18-20'})

# -------------------------------
# Feature Engineering
# -------------------------------
DF['Dandruff'] = DF['Scalp Condition'].apply(
    lambda x: 1 if 'Dandruff' in str(x) else 0
)

DF['Oily_Scalp'] = DF['Scalp Condition'].apply(
    lambda x: 1 if 'Oily' in str(x) else 0
)

DF['Itching'] = DF['Scalp Condition'].apply(
    lambda x: 1 if 'Itching' in str(x) else 0
)

DF.drop(columns=['Scalp Condition'], inplace=True)

# -------------------------------
# Encode categorical columns
# -------------------------------
label_encoders = {}

for col in DF.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    DF[col] = le.fit_transform(DF[col])
    label_encoders[col] = le

# -------------------------------
# Split data
# -------------------------------
TARGET = 'Current Hair Fall Severity'

X = DF.drop(TARGET, axis=1)
y = DF[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
print("Model files saved successfully!")
