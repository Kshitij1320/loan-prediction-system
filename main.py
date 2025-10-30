from src.data_loader import load_loan_data
from src.preprocessor import preprocess_data, split_data
from src.model_trainer import ModelTrainer
import pandas as pd
import joblib


def main():
    print("🚀 Bank Loan Approval Predictor - All Algorithms")
    print("=" * 60)

    # 1. Load and preprocess data
    train_data, test_data = load_loan_data()
    train_processed, encoders = preprocess_data(train_data)

    # FIX: split_data cd "ml project"
    # streamlit run app.pycdreturns 4 values, not 5
    X_train, X_test, y_train, y_test = split_data(train_processed)

    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Testing set: {X_test.shape[0]} samples")

    # 2. Initialize model trainer
    trainer = ModelTrainer()

    # 3. Train all algorithms
    print("\n" + "=" * 50)
    print("TRAINING ALL ALGORITHMS")
    print("=" * 50)

    # Studied algorithms
    print("\n📚 STUDIED ALGORITHMS:")
    lr_acc = trainer.train_linear_regression_scratch(X_train, X_test, y_train, y_test)
    find_s_acc = trainer.train_find_s(X_train, X_test, y_train, y_test)
    knn_acc = trainer.train_knn(X_train, X_test, y_train, y_test)
    rf_acc = trainer.train_random_forest(X_train, X_test, y_train, y_test)

    # New algorithms
    print("\n🔥 NEW ALGORITHMS:")
    new_accuracies = trainer.train_new_algorithms(X_train, X_test, y_train, y_test)

    # 4. Display comparison
    print("\n" + "=" * 50)
    print("📊 ALGORITHM COMPARISON")
    print("=" * 50)

    comparison_table = trainer.get_comparison_table()
    df_comparison = pd.DataFrame(comparison_table)
    print(df_comparison.to_string(index=False))

    # 5. Find best model
    best_algo = max(trainer.results, key=trainer.results.get)
    best_accuracy = trainer.results[best_algo]
    print(f"\n🏆 BEST MODEL: {best_algo.replace('_', ' ').title()}")
    print(f"📈 Best Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")

    # 6. Save the best model
    print("\n💾 Saving the best model...")
    best_model = trainer.models[best_algo]
    joblib.dump(best_model, 'models/best_loan_model.pkl')
    print(f"✅ Best model ({best_algo}) saved to 'models/best_loan_model.pkl'")


if __name__ == "__main__":
    main()