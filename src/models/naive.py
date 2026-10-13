import numpy as np

class NaivePersistence:

    def fit(self, X_train, y_train):
        """
        Nothing to learn.
        """
        return self
    
    def predict(self, X):
        """
        Predict next value as last value in window.
        """
        return X[:, -1]

class DriftNaive:

    def __init__(self):
        self.mean_drift = None

    def fit(self, X_train, y_train):
        """
        Estimate average one-step drift from training windows.
        """

        last_values = X_train[:, -1]

        increments = y_train - last_values

        self.mean_drift = np.mean(increments)

        return self
    
    def predict(self, X):
        """
        Predict next value as lat value + average drift.
        """

        return X[:, -1] + self.mean_drift
    