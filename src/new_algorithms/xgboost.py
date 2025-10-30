try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - install with: pip install xgboost")


class XGBoostModel:
    """XGBoost wrapper"""

    def __init__(self, random_state=42):
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(random_state=random_state)
        else:
            self.model = None

    def fit(self, X, y):
        if self.model:
            self.model.fit(X, y)
        else:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        else:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")


# Test the class
if __name__ == "__main__":
    # Simple test
    import numpy as np

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    try:
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict(X)
        print(f"XGBoost test successful! Predictions: {preds}")
    except ImportError as e:
        print(f"XGBoost test failed: {e}")