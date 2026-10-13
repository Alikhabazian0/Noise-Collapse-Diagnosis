import os
import sys
import numpy as np
import pandas as pd

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

from src.evaluation.collapse import (
    evaluate_by_sigma,
    find_collapse_point
)


def run_single_model_full(
    model,
    model_name: str,
    file_path: str,
    value_column: str = "x",
    train_ratio: float = 0.7,
    save_results: bool = True,
    results_dir: str = "results/2026-06-25_full_benchmark_with_train_predictions"
):
    """
    Run one model on one generated time series.

    Saves:
    1. test-set metrics by sigma
    2. test predictions
    3. full train+test aligned predictions

    Returns:
    evaluation_df
    test_predictions_df
    full_predictions_df
    collapse_sigma
    """

    # ============================
    # Load data
    # ============================

    df = pd.read_csv(file_path)

    series_name = os.path.basename(file_path).replace(".csv", "")

    # ============================
    # Supervised windows
    # ============================

    X_train, X_test, y_train, y_test = prepare_supervised_data(
        df=df,
        value_column=value_column,
        window_length=WINDOW_LENGTH,
        horizon=FORECAST_HORIZON,
        train_ratio=train_ratio
    )

    _, sigma_targets = create_windows(
        values=df["sigma"].values,
        window_length=WINDOW_LENGTH,
        horizon=FORECAST_HORIZON
    )

    split_index = int(len(sigma_targets) * train_ratio)

    sigma_train = sigma_targets[:split_index]
    sigma_test = sigma_targets[split_index:]

    # Target index in the original generated time series.
    target_time_indices = np.arange(
        WINDOW_LENGTH + FORECAST_HORIZON - 1,
        WINDOW_LENGTH + FORECAST_HORIZON - 1 + len(sigma_targets)
    )

    time_train = target_time_indices[:split_index]
    time_test = target_time_indices[split_index:]

    # ============================
    # Fit and predict
    # ============================

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Ensure flat arrays
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    train_preds = np.asarray(train_preds).reshape(-1)
    test_preds = np.asarray(test_preds).reshape(-1)

    # ============================
    # Test evaluation
    # ============================

    evaluation_df = evaluate_by_sigma(
        y_true=y_test,
        y_pred=test_preds,
        sigma_values=sigma_test
    )

    collapse_sigma = find_collapse_point(
        evaluation_df=evaluation_df,
        threshold=COLLAPSE_THRESHOLD
    )

    evaluation_df["model"] = model_name
    evaluation_df["series"] = series_name
    evaluation_df["collapse_sigma"] = collapse_sigma

    # ============================
    # Prediction files
    # ============================

    train_predictions_df = pd.DataFrame({
        "time_index": time_train,
        "split": "train",
        "y_true": y_train,
        "y_pred": train_preds,
        "sigma": sigma_train,
        "model": model_name,
        "series": series_name
    })

    test_predictions_df = pd.DataFrame({
        "time_index": time_test,
        "split": "test",
        "y_true": y_test,
        "y_pred": test_preds,
        "sigma": sigma_test,
        "model": model_name,
        "series": series_name
    })

    full_predictions_df = (
        pd.concat(
            [train_predictions_df, test_predictions_df],
            ignore_index=True
        )
        .sort_values("time_index")
        .reset_index(drop=True)
    )

    # ============================
    # Save outputs
    # ============================

    if save_results:

        metrics_dir = os.path.join(results_dir, "metrics")
        predictions_dir = os.path.join(results_dir, "predictions")
        full_predictions_dir = os.path.join(results_dir, "full_predictions")

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(full_predictions_dir, exist_ok=True)

        metrics_path = os.path.join(
            metrics_dir,
            f"{series_name}_{model_name}_metrics.csv"
        )

        test_preds_path = os.path.join(
            predictions_dir,
            f"{series_name}_{model_name}_predictions.csv"
        )

        full_preds_path = os.path.join(
            full_predictions_dir,
            f"{series_name}_{model_name}_full_predictions.csv"
        )

        evaluation_df.to_csv(metrics_path, index=False)
        test_predictions_df.to_csv(test_preds_path, index=False)
        full_predictions_df.to_csv(full_preds_path, index=False)

        print(f"Saved metrics: {metrics_path}")
        print(f"Saved test predictions: {test_preds_path}")
        print(f"Saved full predictions: {full_preds_path}")

    return (
        evaluation_df,
        test_predictions_df,
        full_predictions_df,
        collapse_sigma
    )