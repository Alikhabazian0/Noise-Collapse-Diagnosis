import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from config.config import(
    WINDOW_LENGTH,
    FORECAST_HORIZON
)

from src.features.windowing import (
    prepare_supervised_data
)

from src.models.naive import (
    NaivePersistence
)

from src.evaluation.metrics import (
    rmse,
    directional_accuracy
)

FILE_PATH = "data/generated/linear_gaussian.csv"

df = pd.read_csv(FILE_PATH)

X_train, X_test, y_train, y_test = prepare_supervised_data(
    df=df,
    value_column="x",
    window_length=WINDOW_LENGTH,
    horizon=FORECAST_HORIZON,
    train_ratio=0.7
)

model = NaivePersistence()

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("RMSE:", rmse(y_test, preds))
print("Directional Accuracy:", directional_accuracy(y_test, preds))
