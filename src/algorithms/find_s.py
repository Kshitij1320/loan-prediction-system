import numpy as np


class FindSAlgorithm:
    """FIND-S Algorithm for concept learning"""

    def __init__(self):
        self.hypothesis = None

    def fit(self, X, y):
        """Train using FIND-S algorithm"""
        print("🧠 Training FIND-S Algorithm...")

        # Convert to numpy arrays
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y

        # Initialize hypothesis with most specific
        self.hypothesis = np.array(['0'] * X_arr.shape[1])

        # Iterate through positive examples
        for i in range(len(X_arr)):
            if y_arr[i] == 1:  # Positive example
                for j in range(len(self.hypothesis)):
                    if self.hypothesis[j] == '0':
                        self.hypothesis[j] = str(X_arr[i][j])
                    elif self.hypothesis[j] != str(X_arr[i][j]):
                        self.hypothesis[j] = '?'

        print(f"✅ Final Hypothesis: {list(self.hypothesis)}")
        return self.hypothesis

    def predict(self, X):
        """Predict using the learned hypothesis"""
        X_arr = X.values if hasattr(X, 'values') else X
        predictions = []

        for sample in X_arr:
            match = True
            for i in range(len(sample)):
                if self.hypothesis[i] != '?' and self.hypothesis[i] != str(sample[i]):
                    match = False
                    break
            predictions.append(1 if match else 0)

        return np.array(predictions)


# Test
if __name__ == "__main__":
    # Simple test
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])

    model = FindSAlgorithm()
    model.fit(X, y)
    preds = model.predict(X)
    print(f"Predictions: {preds}")