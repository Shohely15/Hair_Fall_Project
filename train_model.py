import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# =========================
# Load Dataset
# =========================

df = pd.read_csv("Predicting Hair Loss Severity in University Students with Integrated Lifestyle Analysis in Bangladesh.  (Responses) - Form Responses 1.csv")

# =========================
# Feature Engineering
# =========================

# Handle Scalp Condition

df['Dandruff'] = df['Scalp Condition'].astype(str).apply(
    lambda x: 1 if 'Dandruff' in x else 0
)

df['Oily_Scalp'] = df['Scalp Condition'].astype(str).apply(
    lambda x: 1 if 'Oily' in x else 0
)

df['Itching'] = df['Scalp Condition'].astype(str).apply(
    lambda x: 1 if 'Itching' in x else 0
)

# Remove old column
df.drop(columns=['Scalp Condition'], inplace=True)

# =========================
# Encode Categorical Columns
# =========================

label_encoders = {}

for col in df.select_dtypes(include='object').columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

# =========================
# Split Data
# =========================

TARGET = 'Current Hair Fall Severity'

X = df.drop(TARGET, axis=1)

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Balance Dataset
# =========================

ros = RandomOverSampler(random_state=42)

X_train_resampled, y_train_resampled = ros.fit_resample(
    X_train,
    y_train
)

# =========================
# Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_resampled)

# =========================
# Train Model
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train_resampled)

# =========================
# Save Files
# =========================

pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(scaler, open("scaler.pkl", "wb"))

pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

print("All PKL files created successfully!")
