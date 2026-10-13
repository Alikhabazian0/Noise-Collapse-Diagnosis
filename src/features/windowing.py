import numpy as np
import pandas as pd

def create_windows(
        values: np.ndarray,
        window_length: int,
        horizon: int = 1
):
    """
    Convert a univariate time series into supervised learning windows.
    
    X[i] = values[i : i + window_length]
    y[i] = values[i + window_length + horizon - 1]

    Example with window_length = 16 and horizon =1:
    X = [x_t-15, ..., x_t]
    y = x_t+1
    """

    X = []
    y = []

    n = len(values)

    max_start = n - window_length - horizon + 1

    for start in range(max_start):
        end = start + window_length
        target_index = end + horizon - 1

        X.append(values[start:end])
        y.append(values[target_index])

    X = np.array(X)
    y = np.array(y)

    return X, y

def train_test_split_time_series(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7
):
    """
    Chronological train/test split.
    No shuffling.
    """

    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

def prepare_supervised_data(
        df: pd.DataFrame,
        value_column: str,
        window_length: int,
        horizon: int,
        train_ratio: float = 0.7
):
    """
    Full pipeline:
    DataFrame -> Values -> Windows -> chronological train/test split.
    """

    values = df[value_column].values.astype(float)

    X,y = create_windows(
        values=values,
        window_length=window_length,
        horizon=horizon
    )
    
    X_train, X_test, y_train, y_test = train_test_split_time_series(
        X=X,
        y=y,
        train_ratio = train_ratio
    )

    return X_train, X_test, y_train, y_test