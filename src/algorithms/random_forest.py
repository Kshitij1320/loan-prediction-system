import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestScratch:
    """Random Forest implemented using scikit-learn Decision Trees"""

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """Train multiple decision trees with bootstrapping"""
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y

        np.random.seed(self.random_state)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            n_samples = X_arr.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_arr[indices]
            y_bootstrap = y_arr[indices]

            # Train decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + i
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        """Make predictions using majority voting"""
        X_arr = X.values if hasattr(X, 'values') else X
        all_predictions = []

        for tree in self.trees:
            pred = tree.predict(X_arr)
            all_predictions.append(pred)

        # Majority vote
        all_predictions = np.array(all_predictions)
        final_predictions = []

        for i in range(X_arr.shape[0]):
            votes = all_predictions[:, i]
            most_common = np.bincount(votes).argmax()
            final_predictions.append(most_common)

        return np.array(final_predictions)