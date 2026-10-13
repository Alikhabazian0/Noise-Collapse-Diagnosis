import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.naive import NaivePersistence, DriftNaive
from src.models.arima_model import ARIMAModel
from src.models.xgboost_model import XGBoostModel
from src.models.xgboost_increment_model import XGBoostIncrementModel
from src.models.lstm import LSTMModel
from src.models.lstm_increment import LSTMIncrementModel
from src.models.gru import GRUModel
from src.models.tcn import TCNModel
from src.models.simple_transformer import SimpleCausalTransformerModel
from src.models.residual_transformer import ResidualCausalTransformerModel
from src.models.patchtst import PatchTSTLevelModel
from src.models.causal_encoder_transformer import CausalEncoderTransformerModel

from src.experiments.run_single_model import run_single_model


RESULTS_DIR = "results/2026-06-25_full_benchmark"

SERIES_FILES = [
    "data/generated/linear_gaussian.csv",
    "data/generated/linear_student_t.csv",
    "data/generated/nonlinear_gaussian.csv",
    "data/generated/nonlinear_student_t.csv"
]


def build_models():
    """
    Fresh model instances for each series.
    """

    return [
        ("naive", NaivePersistence()),

        ("drift_naive", DriftNaive()),

        ("ar1", ARIMAModel(order=(1, 0, 0))),

        ("xgboost_level", XGBoostModel()),

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

        ("gru_level", GRUModel(
            hidden_size=32,
            num_layers=1,
            epochs=30,
            batch_size=32,
            learning_rate=0.001
        )),

        ("tcn_level", TCNModel(
            channels=(32, 32),
            kernel_size=3,
            dropout=0.1,
            epochs=30,
            batch_size=32,
            learning_rate=0.001
        )),

        ("simple_causal_transformer", SimpleCausalTransformerModel(
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            epochs=100,
            batch_size=32,
            learning_rate=0.0001
        )),

        ("residual_causal_transformer", ResidualCausalTransformerModel(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            epochs=50,
            batch_size=32,
            learning_rate=0.0003
        )),

        ("causal_encoder_transformer", CausalEncoderTransformerModel(
            d_model=32,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1,
            epochs=50,
            batch_size=32,
            learning_rate=0.0003
        )),

        ("patchtst_level", PatchTSTLevelModel(
            patch_len=4,
            stride=4,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            epochs=50,
            batch_size=32,
            learning_rate=0.0003
        )),
    ]


summary_rows = []

for series_file in SERIES_FILES:

    series_name = os.path.basename(series_file).replace(".csv", "")

    print("\n" + "#" * 100)
    print(f"RUNNING SERIES: {series_name}")
    print("#" * 100)

    models = build_models()

    for model_name, model in models:

        print("\n" + "=" * 100)
        print(f"Running model: {model_name} | series: {series_name}")
        print("=" * 100)

        try:
            evaluation_df, predictions_df, collapse_sigma = run_single_model(
                model=model,
                model_name=model_name,
                file_path=series_file,
                save_results=True,
                results_dir=RESULTS_DIR
            )

            summary_rows.append({
                "series": series_name,
                "model": model_name,
                "collapse_sigma": collapse_sigma if collapse_sigma is not None else "> 2.0",
                "collapse_status": "confirmed" if collapse_sigma is not None else "not confirmed",
                "last_sigma": evaluation_df["sigma"].iloc[-1],
                "last_rmse": evaluation_df["rmse"].iloc[-1],
                "last_rmse_over_sigma": evaluation_df["rmse_over_sigma"].iloc[-1],
                "last_directional_accuracy": evaluation_df["directional_accuracy"].iloc[-1]
            })

        except Exception as e:
            print(f"FAILED: {model_name} on {series_name}")
            print(f"Error: {e}")

            summary_rows.append({
                "series": series_name,
                "model": model_name,
                "collapse_sigma": "ERROR",
                "collapse_status": "ERROR",
                "last_sigma": None,
                "last_rmse": None,
                "last_rmse_over_sigma": None,
                "last_directional_accuracy": None
            })


summary_df = pd.DataFrame(summary_rows)

#os.makedirs("results/summaries", exist_ok=True)
#summary_path = "results/summaries/all_models_all_series_summary.csv"

summary_dir = os.path.join(RESULTS_DIR, "summaries")
os.makedirs(summary_dir, exist_ok=True)

summary_path = os.path.join(
    summary_dir,
    "all_models_all_series_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "#" * 100)
print("FINAL SUMMARY")
print("#" * 100)
print(summary_df)

print(f"\nSaved final summary: {summary_path}")