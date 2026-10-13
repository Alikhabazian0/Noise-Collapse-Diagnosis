import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from config.config import COLLAPSE_THRESHOLD

from src.evaluation.entropy import (
    shannon_entropy,
    permutation_entropy
)

from src.utils.segmentation import (
    get_train_test_split_index,
    get_collapse_index,
    build_precollapse_segments,
    attach_segment_labels
)

RESULTS_DIR = "results/2026-06-25_full_benchmark_with_train_predictions"

FULL_PREDICTIONS_DIR = os.path.join(
    RESULTS_DIR,
    "full_predictions"
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "precollapse_entropy"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SEGMENTS = 10
N_BINS = 30
PERMUTATION_ORDER = 3
PERMUTATION_DELAY = 1


full_predictions_files = glob.glob(
    os.path.join(FULL_PREDICTIONS_DIR, "*_full_predictions.csv")
)

all_rows = []

for full_path in full_predictions_files:

    full_df = pd.read_csv(full_path)

    series = full_df["series"].iloc[0]
    model = full_df["model"].iloc[0]

    metrics_path = os.path.join(
        METRICS_DIR,
        f"{series}_{model}_metrics.csv"
    )

    if not os.path.exists(metrics_path):
        print(f"Skipping missing metrics: {series} | {model}")
        continue

    metrics_df = pd.read_csv(metrics_path)

    train_test_index = get_train_test_split_index(full_df)

    collapse_index = get_collapse_index(
        full_predictions_df=full_df,
        metrics_df=metrics_df,
        collapse_threshold=COLLAPSE_THRESHOLD
    )

    collapse_status = (
        "confirmed"
        if (metrics_df["rmse_over_sigma"] >= COLLAPSE_THRESHOLD).any()
        else "not_confirmed"
    )

    segments_df = build_precollapse_segments(
        full_predictions_df = full_df,
        collapse_index = collapse_index,
        n_segments = N_SEGMENTS
    )

    labeled_df = attach_segment_labels(
        full_predictions_df=full_df,
        segments_df=segments_df
    )

    for _, seg_row in segments_df.iterrows():

        segment_id = int(seg_row["segment"])

        segment_data = labeled_df[
            labeled_df["segment"] == segment_id
        ].copy()

        if segment_data.empty:
            continue

        y_true = segment_data["y_true"].values
        y_pred = segment_data["y_pred"].values

        true_shannon = shannon_entropy(
            y_true,
            n_bins = N_BINS,
            normalize=True
        )

        pred_shannon = shannon_entropy(
            y_pred,
            n_bins=N_BINS,
            normalize=True
        )

        true_permutation = permutation_entropy(
            y_true,
            order = PERMUTATION_ORDER,
            delay = PERMUTATION_DELAY,
            normalize=True
        )

        pred_permutation = permutation_entropy(
            y_pred,
            order = PERMUTATION_ORDER,
            delay = PERMUTATION_DELAY,
            normalize = True
        )

        all_rows.append({
            "series": series,
            "model": model,

            "segment": segment_id,
            "progress_start": seg_row["progress_start"],
            "progress_end": seg_row["progress_end"],
            "progress_mid": seg_row["progress_mid"],

            "start_index": seg_row["start_index"],
            "end_index": seg_row["end_index"],
            "n_samples": seg_row["n_samples"],
            "split_status": seg_row["split_status"],

            "train_test_index": train_test_index,
            "collapse_index": collapse_index,
            "collapse_status": collapse_status,

            "true_shannon_entropy": true_shannon,
            "pred_shannon_entropy": pred_shannon,
            "delta_shannon_entropy": abs(true_shannon - pred_shannon),

            "true_permutation_entropy": true_permutation,
            "pred_permutation_entropy": pred_permutation,
            "delta_permutation_entropy": abs(
                true_permutation - pred_permutation
            )
        })


precollapse_entropy_df = pd.DataFrame(all_rows)

output_path = os.path.join(
    OUTPUT_DIR,
    "precollapse_entropy_by_segment.csv"
)

precollapse_entropy_df.to_csv(output_path, index=False)


summary_df = (
    precollapse_entropy_df
    .groupby(["series", "model"])
    .agg(
        mean_delta_shannon_entropy=("delta_shannon_entropy", "mean"),
        mean_delta_permutation_entropy=("delta_permutation_entropy", "mean"),
        final_delta_shannon_entropy=("delta_shannon_entropy", "last"),
        final_delta_permutation_entropy=("delta_permutation_entropy", "last"),
        mean_true_shannon_entropy=("true_shannon_entropy", "mean"),
        mean_pred_shannon_entropy=("pred_shannon_entropy", "mean"),
        mean_true_permutation_entropy=("true_permutation_entropy", "mean"),
        mean_pred_permutation_entropy=("pred_permutation_entropy", "mean"),
        collapse_index=("collapse_index", "first"),
        train_test_index=("train_test_index", "first"),
        collapse_status=("collapse_status", "first")
    )
    .reset_index()
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "precollapse_entropy_summary.csv"
)

summary_df.to_csv(summary_path, index=False)


print("Saved pre-collapse entropy by segment:")
print(output_path)

print("\nSaved pre-collapse entropy summary:")
print(summary_path)

print("\nPreview:")
print(precollapse_entropy_df.head())