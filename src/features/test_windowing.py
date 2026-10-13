import os 
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.config import WINDOW_LENGTH, FORECAST_HORIZON
from src.features.windowing import prepare_supervised_data

FILE_PATH = "data/generated/linear_gaussian.csv"

df = pd.read_csv(FILE_PATH)

X_train, X_test, y_train, y_test = prepare_supervised_data(
    df=df,
    value_column="x",
    window_length=WINDOW_LENGTH,
    horizon=FORECAST_HORIZON,
    train_ratio=0.7
)

print("window length:", WINDOW_LENGTH)
print("forecast horizon:", FORECAST_HORIZON)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nFirst X_train sample:")
print(X_train[0])

print("\nFirst y_train sample:")
print(y_train[0])