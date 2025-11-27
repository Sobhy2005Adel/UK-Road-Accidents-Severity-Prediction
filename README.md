# CRASH-SEVERITY-CLASSIFIER
Crash Severity Classifier predicts the severity of road accidents (Slight, Serious, Fatal) using a trained stacking machine-learning model. The project includes full preprocessing, SMOTE balancing, and a Streamlit app that allows users to input accident data and view real-time predictions.
# 🚦 Crash Severity Classifier  
Machine Learning Project for Predicting Accident Severity (UK Dataset)

## 📌 Overview  
This project predicts the **severity of road accidents** (Slight – Serious – Fatal) using machine-learning models trained on the UK Road Accident dataset.  
The final deployed model is a **Stacking Classifier** trained on SMOTE-balanced data.

A **Streamlit web app** is included to allow users to input accident details and get a severity prediction.

---

## 🧠 Features  
- ML pipeline with:
  - Cleaning & preprocessing  
  - Label Encoding  
  - Numerical Scaling (Robust Scaler)  
  - SMOTE Oversampling  
  - Stacking Ensemble Classifier  
- Web interface built using **Streamlit**
- Supports:
  - Manual user inputs  
  - Auto-generated random scenario inputs  
- Model interprets severity into:
  - **Slight**
  - **Serious**
  - **Fatal**

---

## 📁 Project Structure

---

## 🚀 How to Run the Streamlit App

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
streamlit
pandas
numpy
scikit-learn
joblib

###2️⃣ Run the App
streamlit run project1_app.py
