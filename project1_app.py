# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pickle
import warnings
warnings.filterwarnings('ignore')
# -----------------------------
# Load Model, Scaler, Encoder
# -----------------------------
model_info = load("stacking_smote_model.joblib")
stack_model = model_info["model"]
feature_names = model_info["features"]

with open("Accident_Severity_label_encoder.pkl", "rb") as f:
    le_target = pickle.load(f)

with open("robust_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Translation mapping for user-friendly target
severity_map = {0: "Slight", 1: "Serious", 2: "Fatal"}  # خفيف / خطير / قاتل

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("UK Road Accidents Severity Prediction 🚦")
st.write("Enter accident and casualty details:")

# Option to use random values
use_random = st.checkbox("Use random values for features")

# جمع مدخلات المستخدم
user_input = {}
for col in feature_names:
    if col in ['Number_of_Casualties', 'Age_of_Casualty', 'Age_of_Driver', 'Speed_limit']:
        if use_random:
            # Random integer within a realistic range
            if col == 'Speed_limit':
                user_input[col] = np.random.choice([20, 30, 40, 50, 60, 70])
            elif col == 'Number_of_Casualties':
                user_input[col] = np.random.randint(1, 5)
            elif col == 'Age_of_Casualty' or col == 'Age_of_Driver':
                user_input[col] = np.random.randint(18, 80)
        else:
            user_input[col] = st.number_input(col, min_value=0, step=1)
    else:
        if use_random:
            # Random string placeholder for categorical features
            user_input[col] = "random_value"
        else:
            user_input[col] = st.text_input(col)

if st.button("Predict Accident Severity"):
    # Prepare Data
    df_input = pd.DataFrame([user_input])

    # Add missing columns if any
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    # Arrange columns
    df_input = df_input[feature_names]

    # Encoding for categorical columns
    for col in df_input.select_dtypes(include='object').columns:
        try:
            with open(f"{col}_label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
                df_input[col] = le.transform(df_input[col])
        except:
            df_input[col] = 0

    # Scale numerical columns
    num_cols = scaler.feature_names_in_
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    # Prediction
    y_pred = stack_model.predict(df_input)
    y_pred_label = le_target.inverse_transform(y_pred)

    # Translate result
    severity_label = severity_map.get(y_pred_label[0], str(y_pred_label[0]))
    st.success(f"Predicted Accident Severity: {severity_label}")

# streamlit run project1_app.py