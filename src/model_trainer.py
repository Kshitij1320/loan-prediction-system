import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Studied algorithms (from scratch)
from src.algorithms.linear_regression_scratch import LinearRegressionScratch
from src.algorithms.find_s import FindSAlgorithm
from src.algorithms.knn import KNNScratch
from src.algorithms.random_forest import RandomForestScratch

# New algorithms
from src.new_algorithms.logistic_regression import LogisticRegressionModel
from src.new_algorithms.svm import SVMModel

# XGBoost with error handling
try:
    from src.new_algorithms.xgboost import XGBoostModel

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available")


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}

    def train_linear_regression_scratch(self, X_train, X_test, y_train, y_test):
        """Train linear regression from scratch"""
        print("🧠 Training Linear Regression (from scratch)...")

        model = LinearRegressionScratch(learning_rate=0.001, n_iterations=1000)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict_binary(X_test.values)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['linear_regression_scratch'] = model
        self.results['linear_regression_scratch'] = accuracy
        return accuracy

    def train_find_s(self, X_train, X_test, y_train, y_test):
        """Train FIND-S algorithm"""
        print("🧠 Training FIND-S Algorithm...")

        model = FindSAlgorithm()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['find_s'] = model
        self.results['find_s'] = accuracy
        return accuracy

    def train_knn(self, X_train, X_test, y_train, y_test):
        """Train K-Nearest Neighbors from scratch"""
        print("🧠 Training K-Nearest Neighbors (from scratch)...")

        model = KNNScratch(k=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['knn'] = model
        self.results['knn'] = accuracy
        return accuracy

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest from scratch"""
        print("🧠 Training Random Forest (from scratch)...")

        model = RandomForestScratch(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['random_forest'] = model
        self.results['random_forest'] = accuracy
        return accuracy

    def train_new_algorithms(self, X_train, X_test, y_train, y_test):
        """Train new algorithms not studied in class"""
        print("🧠 Training New Algorithms...")

        new_accuracies = {}

        # Logistic Regression
        print("📊 Training Logistic Regression...")
        lr_model = LogisticRegressionModel(max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = lr_accuracy
        new_accuracies['logistic_regression'] = lr_accuracy
        print(f"✅ Logistic Regression Accuracy: {lr_accuracy:.4f}")

        # Support Vector Machine
        print("📊 Training SVM...")
        svm_model = SVMModel()
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        self.models['svm'] = svm_model
        self.results['svm'] = svm_accuracy
        new_accuracies['svm'] = svm_accuracy
        print(f"✅ SVM Accuracy: {svm_accuracy:.4f}")

        # XGBoost (only if available)
        if XGBOOST_AVAILABLE:
            print("📊 Training XGBoost...")
            xgb_model = XGBoostModel()
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            self.models['xgboost'] = xgb_model
            self.results['xgboost'] = xgb_accuracy
            new_accuracies['xgboost'] = xgb_accuracy
            print(f"✅ XGBoost Accuracy: {xgb_accuracy:.4f}")
        else:
            print("⏭️  Skipping XGBoost")

        return new_accuracies

    def get_comparison_table(self):
        """Create a comparison table of all algorithms"""
        comparison = []
        for algo, accuracy in self.results.items():
            comparison.append({
                'Algorithm': algo.replace('_', ' ').title(),
                'Accuracy': f"{accuracy:.4f}",
                'Accuracy_Percent': f"{(accuracy * 100):.2f}%"
            })

        # Sort by accuracy descending
        comparison.sort(key=lambda x: float(x['Accuracy']), reverse=True)
        return comparison


# Test the ModelTrainer class
if __name__ == "__main__":
    print("✅ ModelTrainer class is working!")