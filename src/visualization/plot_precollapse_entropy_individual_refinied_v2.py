import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.evaluation.entropy import (
    shannon_entropy,
    permutation_entropy
)


RESULTS_DIR = "results/2026-06-25_full_benchmark_with_train_predictions"

FULL_PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "full_predictions")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

ENTROPY_PATH = os.path.join(
    RESULTS_DIR,
    "precollapse_entropy",
    "precollapse_entropy_by_segment.csv"
)

FIGURE_DIR = "figures/precollapse_entropy/individual_refined_v2"
os.makedirs(FIGURE_DIR, exist_ok=True)

COLLAPSE_THRESHOLD = 1.0

N_BINS = 30
PERMUTATION_ORDER = 3
PERMUTATION_DELAY = 1


def model_label(name):
    return name.replace("_", " ").title()


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        if isinstance(x, str) and ">" in x:
            return None
        return float(x)
    except Exception:
        return None


def get_event_indices(full_df, metrics_df):
    metrics_df = metrics_df.sort_values("sigma").reset_index(drop=True)
    test_df = full_df[full_df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("No test split found.")

    first_sign_rows = metrics_df[
        metrics_df["rmse_over_sigma"] >= COLLAPSE_THRESHOLD
    ]

    if first_sign_rows.empty:
        first_sign_index = None
    else:
        first_sign_sigma = first_sign_rows["sigma"].iloc[0]
        matching = test_df[test_df["sigma"] == first_sign_sigma]
        first_sign_index = int(matching["time_index"].min()) if not matching.empty else None

    confirmed_index = None
    collapse_confirmed = False

    if "collapse_sigma" in metrics_df.columns:
        non_null = metrics_df["collapse_sigma"].dropna()

        if not non_null.empty:
            collapse_sigma = safe_float(non_null.iloc[0])

            if collapse_sigma is not None:
                matching = test_df[test_df["sigma"] == collapse_sigma]

                if not matching.empty:
                    confirmed_index = int(matching["time_index"].min())
                    collapse_confirmed = True

    if confirmed_index is None:
        confirmed_index = int(test_df["time_index"].max())

    return first_sign_index, confirmed_index, collapse_confirmed


def index_to_progress(index_value, start_index, collapse_index):
    if index_value is None:
        return None

    if collapse_index <= start_index:
        return None

    progress = 100 * (index_value - start_index) / (collapse_index - start_index)
    return float(np.clip(progress, 0, 100))


def add_time_event_lines(ax, train_test_index, first_sign_index, confirmed_index, collapse_confirmed):
    ax.axvline(
        train_test_index,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label="Train/Test split"
    )

    if first_sign_index is not None:
        ax.axvline(
            first_sign_index,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="First sign of collapse"
        )

    if collapse_confirmed:
        ax.axvline(
            confirmed_index,
            color="red",
            linestyle=":",
            linewidth=2.3,
            label="Confirmed collapse"
        )


def add_progress_event_lines(ax, train_progress, first_progress, collapse_progress, collapse_confirmed):
    if train_progress is not None:
        ax.axvline(
            train_progress,
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            label="Train/Test split"
        )

    if first_progress is not None:
        ax.axvline(
            first_progress,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="First sign of collapse"
        )

    if collapse_confirmed:
        ax.axvline(
            collapse_progress,
            color="red",
            linestyle=":",
            linewidth=2.3,
            label="Confirmed collapse"
        )


def compute_entropy_on_interval(df, start_index, end_index):
    interval_df = df[
        (df["time_index"] >= start_index)
        & (df["time_index"] <= end_index)
    ].copy()

    y_true = interval_df["y_true"].values
    y_pred = interval_df["y_pred"].values

    true_shannon = shannon_entropy(
        y_true,
        n_bins=N_BINS,
        normalize=True
    )

    pred_shannon = shannon_entropy(
        y_pred,
        n_bins=N_BINS,
        normalize=True
    )

    true_permutation = permutation_entropy(
        y_true,
        order=PERMUTATION_ORDER,
        delay=PERMUTATION_DELAY,
        normalize=True
    )

    pred_permutation = permutation_entropy(
        y_pred,
        order=PERMUTATION_ORDER,
        delay=PERMUTATION_DELAY,
        normalize=True
    )

    return {
        "true_shannon_entropy": true_shannon,
        "pred_shannon_entropy": pred_shannon,
        "delta_shannon_entropy": abs(true_shannon - pred_shannon),
        "true_permutation_entropy": true_permutation,
        "pred_permutation_entropy": pred_permutation,
        "delta_permutation_entropy": abs(true_permutation - pred_permutation),
    }


entropy_df = pd.read_csv(ENTROPY_PATH)

full_prediction_files = glob.glob(
    os.path.join(FULL_PREDICTIONS_DIR, "*_full_predictions.csv")
)

for full_path in full_prediction_files:

    full_df = pd.read_csv(full_path)

    series = full_df["series"].iloc[0]
    model = full_df["model"].iloc[0]

    metrics_path = os.path.join(
        METRICS_DIR,
        f"{series}_{model}_metrics.csv"
    )

    if not os.path.exists(metrics_path):
        print(f"Missing metrics: {series} | {model}")
        continue

    metrics_df = pd.read_csv(metrics_path)

    subset = entropy_df[
        (entropy_df["series"] == series)
        & (entropy_df["model"] == model)
    ].sort_values("segment").copy()

    if subset.empty:
        print(f"Skipping missing entropy: {series} | {model}")
        continue

    train_test_index = int(subset["train_test_index"].iloc[0])

    first_sign_index, confirmed_index, collapse_confirmed = get_event_indices(
        full_df=full_df,
        metrics_df=metrics_df
    )

    start_index = int(subset["start_index"].min())
    collapse_index_for_progress = confirmed_index

    progress = list((subset["progress_mid"] * 100).values)

    # ============================
    # Add explicit 100% point
    # ============================

    boundary_entropy = compute_entropy_on_interval(
        df=full_df,
        start_index=int(subset["start_index"].iloc[-1]),
        end_index=confirmed_index
    )

    progress.append(100.0)

    true_shannon_values = list(subset["true_shannon_entropy"].values)
    pred_shannon_values = list(subset["pred_shannon_entropy"].values)
    delta_shannon_values = list(subset["delta_shannon_entropy"].values)

    true_permutation_values = list(subset["true_permutation_entropy"].values)
    pred_permutation_values = list(subset["pred_permutation_entropy"].values)
    delta_permutation_values = list(subset["delta_permutation_entropy"].values)

    true_shannon_values.append(boundary_entropy["true_shannon_entropy"])
    pred_shannon_values.append(boundary_entropy["pred_shannon_entropy"])
    delta_shannon_values.append(boundary_entropy["delta_shannon_entropy"])

    true_permutation_values.append(boundary_entropy["true_permutation_entropy"])
    pred_permutation_values.append(boundary_entropy["pred_permutation_entropy"])
    delta_permutation_values.append(boundary_entropy["delta_permutation_entropy"])

    train_progress = index_to_progress(
        train_test_index,
        start_index,
        collapse_index_for_progress
    )

    first_progress = index_to_progress(
        first_sign_index,
        start_index,
        collapse_index_for_progress
    )

    collapse_progress = 100.0

    status_text = "CONFIRMED" if collapse_confirmed else "NOT CONFIRMED"
    status_color = "green" if collapse_confirmed else "red"

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 14),
        sharex=False
    )

    # ============================
    # Panel 1: Time series
    # ============================

    axes[0].plot(
        full_df["time_index"],
        full_df["y_true"],
        color="black",
        linewidth=1.4,
        label="True series"
    )

    axes[0].plot(
        full_df["time_index"],
        full_df["y_pred"],
        color="red",
        linewidth=1.0,
        alpha=0.75,
        label="Prediction"
    )

    add_time_event_lines(
        axes[0],
        train_test_index,
        first_sign_index,
        confirmed_index,
        collapse_confirmed
    )

    axes[0].set_title(
        f"{model_label(model)} on {series}",
        fontsize=16,
        fontweight="bold"
    )

    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.3)

    axes[0].text(
        1.02,
        0.88,
        f"{status_text}\nThreshold RMSE/σ = {COLLAPSE_THRESHOLD}\nConsecutive rule = 2",
        transform=axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
        color=status_color,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    axes[0].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.43),
        fontsize=9,
        frameon=True
    )

    # ============================
    # Panel 2: Shannon
    # ============================

    axes[1].plot(
        progress,
        true_shannon_values,
        marker="o",
        color="black",
        linewidth=2.0,
        label="True Shannon"
    )

    axes[1].plot(
        progress,
        pred_shannon_values,
        marker="o",
        color="tab:blue",
        linewidth=2.0,
        alpha=0.85,
        label="Predicted Shannon"
    )

    add_progress_event_lines(
        axes[1],
        train_progress,
        first_progress,
        collapse_progress,
        collapse_confirmed
    )

    axes[1].set_ylabel("Shannon Entropy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)

    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True
    )

    # ============================
    # Panel 3: Permutation
    # ============================

    axes[2].plot(
        progress,
        true_permutation_values,
        marker="o",
        color="black",
        linewidth=2.0,
        label="True Permutation"
    )

    axes[2].plot(
        progress,
        pred_permutation_values,
        marker="o",
        color="green",
        linewidth=2.0,
        alpha=0.85,
        label="Predicted Permutation"
    )

    add_progress_event_lines(
        axes[2],
        train_progress,
        first_progress,
        collapse_progress,
        collapse_confirmed
    )

    axes[2].set_ylabel("Permutation Entropy")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)

    axes[2].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True
    )

    # ============================
    # Panel 4: Delta entropy
    # ============================

    axes[3].plot(
        progress,
        delta_shannon_values,
        marker="o",
        color="tab:orange",
        linewidth=2.0,
        label="Δ Shannon"
    )

    axes[3].plot(
        progress,
        delta_permutation_values,
        marker="o",
        color="purple",
        linewidth=2.0,
        label="Δ Permutation"
    )

    add_progress_event_lines(
        axes[3],
        train_progress,
        first_progress,
        collapse_progress,
        collapse_confirmed
    )

    axes[3].set_xlabel("Progress Toward Collapse (%)")
    axes[3].set_ylabel("Entropy Error")
    axes[3].grid(alpha=0.3)

    axes[3].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True
    )

    xticks = list((subset["progress_mid"] * 100).values) + [100.0]
    axes[3].set_xticks(xticks)
    axes[3].set_xticklabels([f"{int(x)}%" for x in xticks])

    for ax in axes[1:]:
        ax.set_xlim(0, 102)

    plt.tight_layout(rect=[0, 0, 0.83, 1])

    path = os.path.join(
        FIGURE_DIR,
        f"{series}_{model}_precollapse_entropy_refined_v2.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


print("\nFinished refined V2 individual pre-collapse entropy plots.")