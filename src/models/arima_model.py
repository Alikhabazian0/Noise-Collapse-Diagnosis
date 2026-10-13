from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModel:

    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self.history = None

    def fit(self, X_train, y_train):
        """
        Store the training series.
        ARIMA will be refit step-by-step during prediction.
        """
        
        self.history = list(
             np.concatenate([
                  X_train[0],
                  y_train
             ])
        )

        return self
    
    def predict(self, X_test):
        """
        Walk-forward one-step forecasting.

        For each test point:
        1. Fit ARIMA on available history
        2. Forecast one step ahead
        3. Add the true observed value from X_test to history
        """

        predictions = []

        history = self.history.copy()

        for i in range(len(X_test)):

            model = ARIMA(history, order=self.order)
            fitted_model = model.fit()

            forecast = fitted_model.forecast(steps=1)[0]
            predictions.append(forecast)

            true_observed_value = X_test[i, -1]
            history.append(true_observed_value)

        return np.array(predictions)
    