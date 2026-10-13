import numpy as np
from xgboost import XGBRegressor

class XGBoostIncrementModel:

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        """
        Train on incerments:
        target = y_train - last value in window
        """

        last_values = X_train[:, -1]
        y_increment = y_train - last_values

        self.model.fit(X_train, y_increment)

        return self
    
    def predict(self, X_test):
        """
        predict increment, then reconstruct level:
        x_hat[t+1] = x[t] + predicted_increment
        """

        last_values = X_test[:, -1]
        pred_increment = self.model.predict(X_test)

        return last_values + pred_increment