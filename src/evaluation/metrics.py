import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

def directional_accuracy(y_true, y_pred):
    """
    Compare direction relative to previous value.
    """

    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    true_sign = np.sign(true_diff)
    pred_sign = np.sign(pred_diff)

    return np.mean(true_sign == pred_sign)
