import os
import sys
import glob
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.evaluation.entropy import entropy_by_sigma

RESULTS_DIR = "results/2026-06-25_full_benchmark"
PREDICTION_DIR = os.path.join(RESULTS_DIR, "predictions")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "entropy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BINS = 30
PERMUTATION_ORDER = 3
PERMUTATION_DELAY = 1

prediction_files = glob.glob(
    os.path.join(PREDICTION_DIR, "*_predictions.csv")
)

all_rows = []

for file_path in prediction_files:

    df = pd.read_csv(file_path)

    series = df["series"].iloc[0]
    model = df["model"].iloc[0]

    sigma_values = df["sigma"].values

    # true entropy
    true_rows = entropy_by_sigma(
        values=df["y_true"].values,
        sigma_values=sigma_values,
        n_bins=N_BINS,
        order=PERMUTATION_ORDER,
        delay=PERMUTATION_DELAY
    )

    # predicted entropy
    pred_rows = entropy_by_sigma(
        values=df["y_pred"].values,
        sigma_values=sigma_values,
        n_bins=N_BINS,
        order=PERMUTATION_ORDER,
        delay=PERMUTATION_DELAY
    )

    for true_row, pred_row in zip(true_rows, pred_rows):

        all_rows.append({
            "series": series,
            "model": model,
            "sigma": true_row["sigma"],
            "n_samples": true_row["n_samples"],

            "true_shannon_entropy": true_row["shannon_entropy"],
            "pred_shannon_entropy": pred_row["shannon_entropy"],
            "delta_shannon_entropy": abs(
                true_row["shannon_entropy"]
                - pred_row["shannon_entropy"]
            ),

            "true_permutation_entropy": true_row["permutation_entropy"],
            "pred_permutation_entropy": pred_row["permutation_entropy"],
            "delta_permutation_entropy": abs(
                true_row["permutation_entropy"]
                - pred_row["permutation_entropy"]
            )
        })

entropy_df = pd.DataFrame(all_rows)

entropy_path = os.path.join(
    OUTPUT_DIR,
    "entropy_results_by_sigma.csv"
)

entropy_df.to_csv(entropy_path, index=False)

summary_df = (
    entropy_df
    .groupby(["series", "model"])
    .agg(
        mean_delta_shannon_entropy=("delta_shannon_entropy", "mean"),
        mean_delta_permutation_entropy=("delta_permutation_entropy", "mean"),
        last_delta_shannon_entropy=("delta_shannon_entropy", "last"),
        last_delta_permutation_entropy=("delta_permutation_entropy", "last")
    )
    .reset_index()
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "entropy_summary_by_model.csv"
)

summary_df.to_csv(summary_path, index=False)

print("Saved entropy results:")
print(entropy_path)

print("Saved entropy summary:")
print(summary_path)

print("\nPreview:")
print(entropy_df.head())