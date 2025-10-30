import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the trained model
try:
    model = joblib.load('models/best_loan_model.pkl')
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.info("💡 Run main.py first to train and save the model")
    model = None


def preprocess_input(gender, married, dependents, education, employed,
                     applicant_income, coapplicant_income, loan_amount,
                     loan_term, credit_history, property_area):
    """Preprocess user input exactly like training data"""

    # Create a DataFrame with the input (MATCHING TRAINING DATA SCALE)
    input_data = {
        'Gender': [1 if gender == "Male" else 0],
        'Married': [1 if married == "Yes" else 0],
        'Dependents': [0 if dependents == "0" else 1 if dependents == "1" else 2 if dependents == "2" else 3],
        'Education': [0 if education == "Graduate" else 1],
        'Self_Employed': [1 if employed == "Yes" else 0],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [1 if credit_history == "Good" else 0],
        'Property_Area': [0 if property_area == "Urban" else 1 if property_area == "Semiurban" else 2]
    }

    df = pd.DataFrame(input_data)

    # SCALE the numerical features (like during training)
    # Using approximate scaling factors from the training data
    df['ApplicantIncome'] = df['ApplicantIncome'] / 1000
    df['CoapplicantIncome'] = df['CoapplicantIncome'] / 1000
    df['LoanAmount'] = df['LoanAmount'] / 100
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'] / 100

    return df


def main():
    st.set_page_config(page_title="Bank Loan Predictor", layout="centered")

    st.title("🏦 Bank Loan Approval Predictor")
    st.write("Fill in the applicant details to check loan eligibility")

    with st.form("loan_form"):
        st.header("Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

        with col2:
            employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit History", ["Good", "Bad"])

        st.header("Financial Information")
        col3, col4 = st.columns(2)

        with col3:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)

        with col4:
            # NOTE: Loan amounts in original data were around 100-200
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=120)
            loan_term = st.slider("Loan Term (months)", 12, 480, 360)

        submitted = st.form_submit_button("Check Loan Approval")

        if submitted and model is not None:
            # Preprocess input and make prediction
            input_df = preprocess_input(
                gender, married, dependents, education, employed,
                applicant_income, coapplicant_income, loan_amount,
                loan_term, credit_history, property_area
            )

            # DEBUG: Show processed features
            with st.expander("🔍 Debug: Processed Features"):
                st.write(input_df)

            # Make prediction
            try:
                prediction = model.predict(input_df)[0]

                # Handle probability prediction
                try:
                    probability = model.predict_proba(input_df)[0]
                    approval_prob = probability[1] if len(probability) > 1 else probability[0]
                    rejection_prob = probability[0]
                except AttributeError:
                    # Fallback if predict_proba not available
                    approval_prob = 0.8 if prediction == 1 else 0.2
                    rejection_prob = 1 - approval_prob

                # Display results
                st.markdown("---")
                st.header("📊 Loan Approval Result")

                if prediction == 1:
                    st.success(f"✅ **LOAN APPROVED!**")
                    st.metric("Approval Probability", f"{approval_prob:.2%}")
                else:
                    st.error(f"❌ **LOAN REJECTED**")
                    st.metric("Rejection Probability", f"{rejection_prob:.2%}")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.info("The model might be expecting different features")


if __name__ == "__main__":
    main()