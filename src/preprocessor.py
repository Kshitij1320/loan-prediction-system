import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(data):
    """Preprocess the loan data for machine learning"""
    print("🔄 Preprocessing data...")

    # Create a copy
    df = data.copy()

    # Handle missing values first (without inplace warnings)
    print("🔍 Handling missing values...")

    # Numerical columns - fill with median
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']  # ← Removed Credit_History
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical columns - fill with mode
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                        'Self_Employed', 'Property_Area', 'Credit_History']  # ← Added Credit_History
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)

    # Handle categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Convert Loan_Status to binary (Y=1, N=0)
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    print("✅ Data preprocessing completed!")
    print(f"📊 Missing values after preprocessing: {df.isnull().sum().sum()}")
    return df, label_encoders


def split_data(data, test_size=0.2):
    """Split data into train and test sets - RETURNS 4 VALUES"""
    # Drop Loan_ID as it's not useful for prediction
    X = data.drop(['Loan_Status', 'Loan_ID'], axis=1, errors='ignore')
    y = data['Loan_Status']

    # Scale the features for better algorithm performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    # Return only 4 values, not 5
    return X_train, X_test, y_train, y_test


# Test function
if __name__ == "__main__":
    from data_loader import load_loan_data

    data = load_loan_data()
    processed_data, encoders = preprocess_data(data)
    print("Processed data shape:", processed_data.shape)
    print("Missing values:", processed_data.isnull().sum().sum())