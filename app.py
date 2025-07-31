import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Title and description
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease using a Random Forest model.")

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

# Train the model
@st.cache_resource
def train_model(df):
    # Features and target
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Encode categorical variables
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, X_encoded.columns

# Load data
df = load_data()

# Check if model exists, else train it
model_path = "rf_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    feature_names = pd.get_dummies(df.drop("HeartDisease", axis=1), 
                                  columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], 
                                  drop_first=True).columns
else:
    model, feature_names = train_model(df)

# Create input form
st.subheader("Enter Patient Details")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120, step=1)
        cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=600, value=200, step=1)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=220, value=150, step=1)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    submitted = st.form_submit_button("Predict")

# Process input and make prediction
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain_type],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    # Encode categorical variables
    input_encoded = pd.get_dummies(input_data, columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], drop_first=True)

    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training data
    input_encoded = input_encoded[feature_names]

    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("The model predicts a **high risk** of heart disease.")
        st.write(f"Probability of heart disease: **{probability:.2%}**")
    else:
        st.success("The model predicts a **low risk** of heart disease.")
        st.write(f"Probability of heart disease: **{probability:.2%}**")

# Display dataset info
st.subheader("About the Model")
st.write("This app uses a Random Forest Classifier trained on the heart disease dataset with 303 records and 13 features. The model predicts the likelihood of heart disease based on patient attributes.")

# Optional: Display dataset preview
if st.checkbox("Show dataset preview"):
    st.write(df.head())