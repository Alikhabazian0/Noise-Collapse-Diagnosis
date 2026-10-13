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

from src.experiments.run_single_model_full import run_single_model_full


RESULTS_DIR = "results/2026-06-25_full_benchmark_with_train_predictions"

SERIES_FILES = [
    "data/generated/linear_gaussian.csv",
    "data/generated/linear_student_t.csv",
    "data/generated/nonlinear_gaussian.csv",
    "data/generated/nonlinear_student_t.csv"
]


def build_models():

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


def extract_corrected_summary(evaluation_df, collapse_sigma):

    evaluation_df = evaluation_df.sort_values("sigma").reset_index(drop=True)

    max_sigma_row = evaluation_df.iloc[-1]

    if collapse_sigma is None:
        collapse_status = "not confirmed"
        collapse_sigma_value = "> max_sigma"

        collapse_row = max_sigma_row.copy()
        pre_collapse_df = evaluation_df.copy()

    else:
        collapse_status = "confirmed"
        collapse_sigma_value = collapse_sigma

        collapse_row = evaluation_df[
            evaluation_df["sigma"] == collapse_sigma
        ].iloc[0]

        pre_collapse_df = evaluation_df[
            evaluation_df["sigma"] <= collapse_sigma
        ].copy()

    return {
        "collapse_sigma": collapse_sigma_value,
        "collapse_status": collapse_status,

        "collapse_rmse": collapse_row["rmse"],
        "collapse_rmse_over_sigma": collapse_row["rmse_over_sigma"],
        "collapse_directional_accuracy": collapse_row["directional_accuracy"],

        "precollapse_mean_rmse": pre_collapse_df["rmse"].mean(),
        "precollapse_mean_rmse_over_sigma": pre_collapse_df["rmse_over_sigma"].mean(),
        "precollapse_mean_directional_accuracy": pre_collapse_df["directional_accuracy"].mean(),

        "max_sigma": max_sigma_row["sigma"],
        "max_sigma_rmse": max_sigma_row["rmse"],
        "max_sigma_rmse_over_sigma": max_sigma_row["rmse_over_sigma"],
        "max_sigma_directional_accuracy": max_sigma_row["directional_accuracy"],
    }


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
            (
                evaluation_df,
                test_predictions_df,
                full_predictions_df,
                collapse_sigma
            ) = run_single_model_full(
                model=model,
                model_name=model_name,
                file_path=series_file,
                save_results=True,
                results_dir=RESULTS_DIR
            )

            corrected_metrics = extract_corrected_summary(
                evaluation_df=evaluation_df,
                collapse_sigma=collapse_sigma
            )

            summary_rows.append({
                "series": series_name,
                "model": model_name,
                **corrected_metrics
            })

        except Exception as e:

            print(f"FAILED: {model_name} on {series_name}")
            print(f"Error: {e}")

            summary_rows.append({
                "series": series_name,
                "model": model_name,
                "collapse_sigma": "ERROR",
                "collapse_status": "ERROR",

                "collapse_rmse": None,
                "collapse_rmse_over_sigma": None,
                "collapse_directional_accuracy": None,

                "precollapse_mean_rmse": None,
                "precollapse_mean_rmse_over_sigma": None,
                "precollapse_mean_directional_accuracy": None,

                "max_sigma": None,
                "max_sigma_rmse": None,
                "max_sigma_rmse_over_sigma": None,
                "max_sigma_directional_accuracy": None,
            })


summary_df = pd.DataFrame(summary_rows)

summary_dir = os.path.join(RESULTS_DIR, "summaries")
os.makedirs(summary_dir, exist_ok=True)

summary_path = os.path.join(
    summary_dir,
    "all_models_all_series_corrected_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "#" * 100)
print("FINAL CORRECTED SUMMARY")
print("#" * 100)
print(summary_df)

print(f"\nSaved corrected summary: {summary_path}")