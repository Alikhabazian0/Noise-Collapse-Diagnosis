import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.naive import NaivePersistence, DriftNaive
from src.models.xgboost_increment_model import XGBoostIncrementModel
from src.models.lstm import LSTMModel
from src.models.lstm_increment import LSTMIncrementModel
from src.experiments.run_single_model import run_single_model


FILE_PATH = "data/generated/nonlinear_gaussian.csv"

MODELS = [
    ("naive", NaivePersistence()),
    ("drift_naive", DriftNaive()),
    ("xgboost_increment", XGBoostIncrementModel()),
    ("lstm_level", LSTMModel(
        hidden_size=32,
        num_layers=1,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )),
    ("lstm_increment", LSTMIncrementModel(
        hidden_size=32,
        num_layers=1,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )),
]

summary_rows = []

for model_name, model in MODELS:

    print("\n" + "=" * 80)
    print(f"Running {model_name} on nonlinear_gaussian")
    print("=" * 80)

    evaluation_df, predictions_df, collapse_sigma = run_single_model(
        model=model,
        model_name=model_name,
        file_path=FILE_PATH,
        save_results=True
    )

    summary_rows.append({
        "series": "nonlinear_gaussian",
        "model": model_name,
        "collapse_sigma": collapse_sigma if collapse_sigma is not None else "> 2.0",
        "collapse_status": "confirmed" if collapse_sigma is not None else "not confirmed",
        "last_rmse": evaluation_df["rmse"].iloc[-1],
        "last_rmse_over_sigma": evaluation_df["rmse_over_sigma"].iloc[-1],
        "last_directional_accuracy": evaluation_df["directional_accuracy"].iloc[-1]
    })

summary_df = pd.DataFrame(summary_rows)

os.makedirs("results/summaries", exist_ok=True)

summary_path = "results/summaries/nonlinear_gaussian_current_models_summary.csv"

summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(summary_df)

print(f"\nSaved: {summary_path}")