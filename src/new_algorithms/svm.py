from sklearn.svm import SVC


class SVMModel:
    """Support Vector Machine wrapper"""

    def __init__(self, random_state=42):
        self.model = SVC(random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)