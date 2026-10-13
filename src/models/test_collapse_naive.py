import os
import sys
import pandas as pd
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from config.config import (
    WINDOW_LENGTH,
    FORECAST_HORIZON,
    COLLAPSE_THRESHOLD
)

from src.features.windowing import (
    prepare_supervised_data,
    create_windows
)

from src.models.naive import NaivePersistence

from src.evaluation.collapse import (
    evaluate_by_sigma,
    find_collapse_point
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

# Create matching sigma windows
# We need sigma value corresponding to each y target.
_, sigma_targets = create_windows(
    values=df["sigma"].values,
    window_length=WINDOW_LENGTH,
    horizon=FORECAST_HORIZON
)

split_index = int(len(sigma_targets) * 0.7)

sigma_test = sigma_targets[split_index:]

model = NaivePersistence()
model.fit(X_train, y_train)

preds = model.predict(X_test)

evaluation_df = evaluate_by_sigma(
    y_true=y_test,
    y_pred=preds,
    sigma_values=sigma_test
)

collapse_sigma = find_collapse_point(
    evaluation_df = evaluation_df,
    threshold=COLLAPSE_THRESHOLD
)

print("\nEvaluation by sigma:")
print(evaluation_df)

print("\nCollapse threshold:", COLLAPSE_THRESHOLD)
print("collapse sigma:", collapse_sigma)