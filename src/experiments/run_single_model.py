import os
import sys
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

def run_single_model(
        model,
        model_name: str,
        file_path: str,
        value_column: str = "x",
        train_ratio: float = 0.7,
        save_results: bool = True,
        results_dir="results"
):
    """
    Run one model on one generated time series.

    Outputs:
    - predictions
    - evaluation by sigma
    - collapse sigma
    """

    df = pd.read_csv(file_path)

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
    sigma_test = sigma_targets[split_index:]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    evaluation_df = evaluate_by_sigma(
        y_true=y_test,
        y_pred=preds,
        sigma_values=sigma_test
    )

    collapse_sigma = find_collapse_point(
        evaluation_df=evaluation_df,
        threshold=COLLAPSE_THRESHOLD
    )

    series_name = os.path.basename(file_path).replace(".csv", "")

    evaluation_df["model"] = model_name
    evaluation_df["series"] = series_name
    evaluation_df["collapse_sigma"] = collapse_sigma

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": preds,
        "sigma": sigma_test,
        "model": model_name,
        "series": series_name
    })

    if save_results:
        
        metrics_dir = os.path.join(results_dir, "metrics")
        predictions_dir = os.path.join(results_dir, "predictions")
#        os.makedirs("results/metrics", exist_ok=True)
#        os.makedirs("results/predictions", exist_ok=True)


#        metrics_path = f"results/metrics/{series_name}_{model_name}_metrics.csv"
#        preds_path = f"results/predictions/{series_name}_{model_name}_predictions.csv"

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        
        metrics_path = os.path.join(
             metrics_dir,
             f"{series_name}_{model_name}_metrics.csv"
             )
        
        preds_path = os.path.join(
            predictions_dir,
            f"{series_name}_{model_name}_predictions.csv"
            )

        evaluation_df.to_csv(metrics_path, index=False)
        predictions_df.to_csv(preds_path, index=False)

        print(f"Saved metrics: {metrics_path}")
        print(f"Saved predictions: {preds_path}")

    return evaluation_df, predictions_df, collapse_sigma
