import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# title
st.title("Titanic Dataset Explorer and Logistic Regression Model")

# Upload Titanic Dataset
uploaded_file = st.file_uploader("Upload the Titanic Dataset (CSV format only)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the Titanic dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Loaded Successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(df)
        # Ensure it is the Titanic dataset by checking specific columns
        required_columns = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
        if not required_columns.issubset(df.columns):
            st.error("This is not the Titanic dataset! Please upload the correct file.")
        else:
            # Data Preprocessing
            st.subheader("Data Preprocessing")
            df['Age'].fillna(df['Age'].median(), inplace=True)
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
            df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
            df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

            # Feature selection
            X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
            y = df["Survived"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the Logistic Regression model
            st.subheader("Training Logistic Regression Model")
            model = LogisticRegression()
            model.fit(X_train, y_train)
            st.write("Model trained successfully!")

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # Performance Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_prob)

            # Checkboxes for optional visualizations and metrics
            if st.checkbox("Show Model Performance Metrics"):
                st.subheader("Performance Metrics")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")
                st.write(f"ROC-AUC Score: {roc_auc:.2f}")

            if st.checkbox("Show Confusion Matrix"):
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                fig, ax = plt.subplots()
                disp.plot(ax=ax)
                st.pyplot(fig)

            if st.checkbox("Show ROC Curve"):
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(roc_auc))
                ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="best")
                st.pyplot(fig)

            if st.checkbox("Visualize Dataset"):
                st.subheader("Dataset Visualizations")

                st.write("Passenger Class Distribution:")
                fig, ax = plt.subplots()
                sns.countplot(x="Pclass", data=df, palette="pastel", ax=ax)
                ax.set_title("Passenger Class Distribution")
                st.pyplot(fig)

                st.write("Gender Distribution:")
                fig, ax = plt.subplots()
                sns.countplot(x="Sex", data=df, palette="coolwarm", ax=ax)
                ax.set_title("Gender Distribution")
                st.pyplot(fig)

                st.write("Survival Rate by Class and Gender:")
                survival_rate = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
                fig, ax = plt.subplots()
                survival_rate.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
                ax.set_title("Survival Rate by Class and Gender")
                st.pyplot(fig)

            # Prediction on user input
            st.sidebar.subheader("Make Predictions")
            st.sidebar.write("Enter the passenger details below to predict survival:")
            Pclass = st.sidebar.selectbox("Pclass (Ticket Class)", [1, 2, 3])
            Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
            Age = st.sidebar.slider("Age", 0, 80, 25)
            SibSp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
            Parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
            Fare = st.sidebar.slider("Fare ($)", 0.0, 600.0, 32.2)
            Embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

            # Encode user input
            Sex = 1 if Sex == "Male" else 0
            Embarked = {"C": 0, "Q": 1, "S": 2}[Embarked]

            # Create input for prediction
            user_input = [[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]]

            # predict only on button click
            if st.sidebar.button("Predict Result"):
                prediction = model.predict(user_input)
                prediction_prob = model.predict_proba(user_input)

                # Display Prediction
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.success(f"The passenger is likely to survive (Probability: {prediction_prob[0][1]:.2f}).")
                else:
                    st.error(f"The passenger is not likely to survive (Probability: {prediction_prob[0][0]:.2f}).")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload the Titanic dataset to proceed.")


