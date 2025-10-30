import numpy as np
from collections import Counter


class KNNScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, 'values') else X
        self.y_train = y.values if hasattr(y, 'values') else y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X_test = X.values if hasattr(X, 'values') else X
        predictions = []

        for test_point in X_test:
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))

            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_labels = [label for (_, label) in k_nearest]
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)