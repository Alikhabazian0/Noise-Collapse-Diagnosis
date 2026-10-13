import os
import sys
import pandas as pd
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from config.config import (
    WINDOW_LENGTH,
    FORECAST_HORIZON,
    COLLAPSE_THRESHOLD
)

from src.models.feature_transformers import FeatureCausalTransformerModel

from src.evaluation.collapse import (
    evaluate_by_sigma,
    find_collapse_point
)


RESULTS_DIR = "results/pe_transformer_experiment/pe_feature_runs"

FEATURE_DIR = "results/pe_transformer_experiment/pe_features"

SERIES_FILES = [
    os.path.join(FEATURE_DIR, "linear_gaussian_with_pe_features.csv"),
    os.path.join(FEATURE_DIR, "linear_student_t_with_pe_features.csv"),
    os.path.join(FEATURE_DIR, "nonlinear_gaussian_with_pe_features.csv"),
    os.path.join(FEATURE_DIR, "nonlinear_student_t_with_pe_features.csv"),
]

FEATURE_SETS = {
    "sigma_feature": ["x", "sigma"],
    "sigma_pe_feature": ["x", "sigma", "pe_feature"],
}


def prepare_multifeature_supervised_data(
    df,
    feature_columns,
    target_column="x",
    sigma_column="sigma",
    window_length=16,
    horizon=1,
    train_ratio=0.7
):
    features = df[feature_columns].values.astype(float)
    target = df[target_column].values.astype(float)
    sigma = df[sigma_column].values.astype(float)

    X = []
    y = []
    sigma_targets = []

    max_start = len(df) - window_length - horizon + 1

    for start in range(max_start):
        end = start + window_length
        target_index = end + horizon - 1

        X.append(features[start:end])
        y.append(target[target_index])
        sigma_targets.append(sigma[target_index])

    X = np.asarray(X)
    y = np.asarray(y)
    sigma_targets = np.asarray(sigma_targets)

    split_index = int(len(X) * train_ratio)

    return (
        X[:split_index],
        X[split_index:],
        y[:split_index],
        y[split_index:],
        sigma_targets[:split_index],
        sigma_targets[split_index:]
    )


def build_model(model_name, input_size):

    if model_name == "simple_causal_transformer":
        return FeatureCausalTransformerModel(
            input_size=input_size,
            residual=False,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.1,
            epochs=30,
            batch_size=32,
            learning_rate=0.001
        )

    if model_name == "residual_causal_transformer":
        return FeatureCausalTransformerModel(
            input_size=input_size,
            residual=True,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            epochs=50,
            batch_size=32,
            learning_rate=0.0003
        )

    if model_name == "causal_encoder_transformer":
        return FeatureCausalTransformerModel(
            input_size=input_size,
            residual=False,
            d_model=32,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1,
            epochs=50,
            batch_size=32,
            learning_rate=0.0003
        )

    raise ValueError(f"Unknown model: {model_name}")


def corrected_summary_from_evaluation(evaluation_df, collapse_sigma):

    evaluation_df = evaluation_df.sort_values("sigma").reset_index(drop=True)

    max_sigma_row = evaluation_df.iloc[-1]

    if collapse_sigma is None:
        collapse_status = "not_confirmed"
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


MODELS = [
    "simple_causal_transformer",
    "residual_causal_transformer",
    "causal_encoder_transformer",
]

metrics_dir = os.path.join(RESULTS_DIR, "metrics")
predictions_dir = os.path.join(RESULTS_DIR, "predictions")
summary_dir = os.path.join(RESULTS_DIR, "summaries")

os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

summary_rows = []

for file_path in SERIES_FILES:

    df = pd.read_csv(file_path)
    series_name = df["series"].iloc[0]

    for experiment_type, feature_columns in FEATURE_SETS.items():

        print("\n" + "#" * 100)
        print(f"SERIES: {series_name} | EXPERIMENT: {experiment_type}")
        print(f"FEATURES: {feature_columns}")
        print("#" * 100)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            sigma_train,
            sigma_test
        ) = prepare_multifeature_supervised_data(
            df=df,
            feature_columns=feature_columns,
            target_column="x",
            sigma_column="sigma",
            window_length=WINDOW_LENGTH,
            horizon=FORECAST_HORIZON,
            train_ratio=0.7
        )

        for model_name in MODELS:

            print("\n" + "=" * 100)
            print(f"Running {model_name} | {series_name} | {experiment_type}")
            print("=" * 100)

            try:
                model = build_model(
                    model_name=model_name,
                    input_size=len(feature_columns)
                )

                val_fraction = 0.2
                val_start = int(len(X_train) * (1 - val_fraction))

                X_fit = X_train[:val_start]
                y_fit = y_train[:val_start]

                X_val = X_train[val_start:]
                y_val = y_train[val_start:]

                model.fit(
                    X_fit,
                    y_fit,
                    X_val=X_val,
                    y_val=y_val
                )

                preds = model.predict(X_test)
                

                # save training history

                history_df = pd.DataFrame(model.history_)
                history_df["series"] = series_name
                history_df["model"] = model_name
                history_df["experiment_type"] = experiment_type
                history_df["feature_columns"] = ",".join(feature_columns)

                history_dir = os.path.join(RESULTS_DIR, "training_history")
                os.makedirs(history_dir, exist_ok=True)

                history_path = os.path.join(
                    history_dir,
                    f"{series_name}_{model_name}_{experiment_type}_history.csv"
                )

                history_df.to_csv(history_path, index=False)
                print(f"Saved training history: {history_path}")

                # evaluation

                evaluation_df = evaluate_by_sigma(
                    y_true=y_test,
                    y_pred=preds,
                    sigma_values=sigma_test
                )

                collapse_sigma = find_collapse_point(
                    evaluation_df=evaluation_df,
                    threshold=COLLAPSE_THRESHOLD
                )

                evaluation_df["series"] = series_name
                evaluation_df["model"] = model_name
                evaluation_df["experiment_type"] = experiment_type
                evaluation_df["feature_columns"] = ",".join(feature_columns)
                evaluation_df["collapse_sigma"] = collapse_sigma

                predictions_df = pd.DataFrame({
                    "y_true": y_test,
                    "y_pred": preds,
                    "sigma": sigma_test,
                    "series": series_name,
                    "model": model_name,
                    "experiment_type": experiment_type,
                    "feature_columns": ",".join(feature_columns)
                })

                metrics_path = os.path.join(
                    metrics_dir,
                    f"{series_name}_{model_name}_{experiment_type}_metrics.csv"
                )

                predictions_path = os.path.join(
                    predictions_dir,
                    f"{series_name}_{model_name}_{experiment_type}_predictions.csv"
                )

                evaluation_df.to_csv(metrics_path, index=False)
                predictions_df.to_csv(predictions_path, index=False)

                corrected_metrics = corrected_summary_from_evaluation(
                    evaluation_df=evaluation_df,
                    collapse_sigma=collapse_sigma
                )

                summary_rows.append({
                    "series": series_name,
                    "model": model_name,
                    "experiment_type": experiment_type,
                    "feature_columns": ",".join(feature_columns),
                    "input_size": len(feature_columns),
                    **corrected_metrics
                })

                print(f"Saved metrics: {metrics_path}")
                print(f"Saved predictions: {predictions_path}")

            except Exception as e:

                print(f"FAILED: {model_name} | {series_name} | {experiment_type}")
                print(f"Error: {e}")

                summary_rows.append({
                    "series": series_name,
                    "model": model_name,
                    "experiment_type": experiment_type,
                    "feature_columns": ",".join(feature_columns),
                    "input_size": len(feature_columns),
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

summary_path = os.path.join(
    summary_dir,
    "pe_feature_transformer_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "#" * 100)
print("PE FEATURE TRANSFORMER EXPERIMENT FINISHED")
print("#" * 100)
print(summary_df)
print(f"\nSaved summary: {summary_path}")