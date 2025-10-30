import pandas as pd
import os


def load_loan_data():
    """Load loan data from CSV files"""
    try:
        # Try to load separate train/test CSV files
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
        print("✅ Separate train/test CSV files loaded")

        # Show data info
        print(f"📊 Train data shape: {train_data.shape}")
        print(f"🔍 Missing values in train data:")
        print(train_data.isnull().sum())

        return train_data, test_data

    except FileNotFoundError:
        print("❌ CSV files not found in data/ folder")
        print("💡 Please make sure you have 'train.csv' and 'test.csv' in data/ folder")

        # Fallback to sample data
        data = create_sample_data()
        return data, None


def create_sample_data():
    """Create sample data if no files found"""
    data = pd.DataFrame({
        'Loan_ID': ['LP001003', 'LP001005', 'LP001006', 'LP001008'],
        'Gender': ['Male', 'Male', 'Male', 'Male'],
        'Married': ['Yes', 'Yes', 'No', 'No'],
        'Dependents': ['1', '0', '0', '0'],
        'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
        'Self_Employed': ['No', 'Yes', 'No', 'No'],
        'ApplicantIncome': [4583, 3000, 2583, 6000],
        'CoapplicantIncome': [1508, 0, 2358, 0],
        'LoanAmount': [128, 66, 120, 141],
        'Loan_Amount_Term': [360, 360, 360, 360],
        'Credit_History': [1, 1, 1, 1],
        'Property_Area': ['Rural', 'Urban', 'Urban', 'Urban'],
        'Loan_Status': ['N', 'Y', 'Y', 'Y']
    })
    print("✅ Sample data created successfully!")
    return data