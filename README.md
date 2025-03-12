# WebApp-Project
This repository is about an application to make predictions on Trained Logistic Regression Model of Titanic Dataset.

##  🚢 Titanic Survival Prediction – Streamlit App
### 📌 Overview
    This Streamlit Web App allows users to:
    ✅ Upload the Titanic dataset (CSV format)
    ✅ Perform data preprocessing and train a Logistic Regression model
    ✅ View key model performance metrics
    ✅ Visualize dataset insights with interactive plots
    ✅ Predict whether a passenger would survive based on user input

### 🎯 Features
    🔹 Upload Titanic Dataset: Accepts only the Titanic dataset and verifies its columns.
    🔹 Data Preprocessing: Handles missing values and encodes categorical features.
    🔹 Train Logistic Regression Model: Automatically trains the model after preprocessing.
    🔹 Performance Metrics (Optional): Displays accuracy, precision, recall, F1-score, and ROC-AUC score.
    🔹 Visualizations (Optional):
     * Confusion Matrix
     * ROC Curve
     * Dataset insights (Passenger Class, Gender Distribution, Survival Rate by Class & Gender)
         🔹 User Input Prediction: Allows users to input passenger details and predict survival.

### 📊 Visualizations
    The app provides interactive visualizations using Matplotlib and Seaborn with enhanced color themes:
    ✅ Confusion Matrix – Heatmap to analyze classification results
    ✅ ROC Curve – Visualizing the trade-off between true positive rate & false positive rate
    ✅ Survival Rate Charts – Bar charts for Passenger Class, Gender Distribution, and Class-Gender Survival Rate

### 🚀 Deployment
    This app can be deployed on Streamlit Cloud or any hosting platform that supports Streamlit apps.

### ✨ Future Improvements
    🚀 Add more ML models for comparison
    🚀 Implement feature selection & hyperparameter tuning
    🚀 Enhance UI with Streamlit components



