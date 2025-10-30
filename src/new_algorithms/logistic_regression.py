from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """Logistic Regression wrapper"""

    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """Add this method for probability predictions"""
        return self.model.predict_proba(X)