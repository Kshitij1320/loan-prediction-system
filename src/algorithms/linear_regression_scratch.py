import numpy as np


class LinearRegressionScratch:
    """Linear Regression implemented from scratch"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict_binary(self, X, threshold=0.5):
        """For classification - convert to binary using threshold"""
        linear_output = self.predict(X)
        return (linear_output >= threshold).astype(int)


# Test the implementation
if __name__ == "__main__":
    # Simple test
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(np.array([[5]]))
    print(f"Prediction for 5: {predictions[0]}")