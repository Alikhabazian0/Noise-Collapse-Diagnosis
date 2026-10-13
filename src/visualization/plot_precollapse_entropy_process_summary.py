import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)


RESULTS_DIR = "results/2026-06-25_full_benchmark_with_train_predictions"

ENTROPY_PATH = os.path.join(
    RESULTS_DIR,
    "precollapse_entropy",
    "precollapse_entropy_by_segment.csv"
)

FIGURE_DIR = "figures/precollapse_entropy/process_summary_all_models"
os.makedirs(FIGURE_DIR, exist_ok=True)


MODEL_ORDER = [
    "naive",
    "drift_naive",
    "ar1",
    "xgboost_level",
    "xgboost_increment",
    "lstm_level",
    "lstm_increment",
    "gru_level",
    "tcn_level",
    "simple_causal_transformer",
    "residual_causal_transformer",
    "causal_encoder_transformer",
    "patchtst_level"
]

SERIES_ORDER = [
    "linear_gaussian",
    "linear_student_t",
    "nonlinear_gaussian",
    "nonlinear_student_t"
]


def model_label(name):
    return name.replace("_", " ").title()


def series_label(name):
    return name.replace("_", " ").title()


def plot_metric_for_series(
    df,
    series,
    y_columns,
    labels,
    title,
    ylabel,
    filename,
    ylim=None
):
    subset = df[df["series"] == series].copy()

    plt.figure(figsize=(14, 6))

    # Plot true reference only once when available
    if "true_shannon_entropy" in y_columns:
        true_col = "true_shannon_entropy"
    elif "true_permutation_entropy" in y_columns:
        true_col = "true_permutation_entropy"
    else:
        true_col = None

    if true_col is not None:
        true_curve = (
            subset
            .groupby("progress_mid")[true_col]
            .mean()
            .reset_index()
            .sort_values("progress_mid")
        )

        plt.plot(
            true_curve["progress_mid"] * 100,
            true_curve[true_col],
            color="black",
            linewidth=3.0,
            linestyle="--",
            label="True Series"
        )

    for model in MODEL_ORDER:

        model_df = (
            subset[subset["model"] == model]
            .sort_values("progress_mid")
        )

        if model_df.empty:
            continue

        for y_col, label_suffix in zip(y_columns, labels):

            # Do not plot true column repeatedly for every model
            if y_col.startswith("true_"):
                continue

            plt.plot(
                model_df["progress_mid"] * 100,
                model_df[y_col],
                marker="o",
                linewidth=1.6,
                alpha=0.85,
                label=f"{model_label(model)}"
            )

    plt.title(
        title,
        fontsize=17,
        fontweight="bold"
    )

    plt.xlabel("Progress Toward Collapse (%)")
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xticks(
        ticks=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        labels=[f"{x}%" for x in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]]
    )

    plt.grid(alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=8,
        frameon=True,
        ncol=1
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


entropy_df = pd.read_csv(ENTROPY_PATH)


for series in SERIES_ORDER:

    # 1. Shannon entropy
    plot_metric_for_series(
        df=entropy_df,
        series=series,
        y_columns=[
            "true_shannon_entropy",
            "pred_shannon_entropy"
        ],
        labels=[
            "True Shannon",
            "Predicted Shannon"
        ],
        title=f"Pre-Collapse Shannon Entropy Evolution: {series_label(series)}",
        ylabel="Normalized Shannon Entropy",
        filename=f"{series}_shannon_entropy_all_models.png",
        ylim=(0, 1.05)
    )

    # 2. Permutation entropy
    plot_metric_for_series(
        df=entropy_df,
        series=series,
        y_columns=[
            "true_permutation_entropy",
            "pred_permutation_entropy"
        ],
        labels=[
            "True Permutation",
            "Predicted Permutation"
        ],
        title=f"Pre-Collapse Permutation Entropy Evolution: {series_label(series)}",
        ylabel="Normalized Permutation Entropy",
        filename=f"{series}_permutation_entropy_all_models.png",
        ylim=(0, 1.05)
    )

    # 3. Delta Shannon
    plot_metric_for_series(
        df=entropy_df,
        series=series,
        y_columns=[
            "delta_shannon_entropy"
        ],
        labels=[
            "Δ Shannon"
        ],
        title=f"Pre-Collapse Shannon Entropy Error: {series_label(series)}",
        ylabel="|True Shannon - Predicted Shannon|",
        filename=f"{series}_delta_shannon_entropy_all_models.png",
        ylim=None
    )

    # 4. Delta Permutation
    plot_metric_for_series(
        df=entropy_df,
        series=series,
        y_columns=[
            "delta_permutation_entropy"
        ],
        labels=[
            "Δ Permutation"
        ],
        title=f"Pre-Collapse Permutation Entropy Error: {series_label(series)}",
        ylabel="|True Permutation - Predicted Permutation|",
        filename=f"{series}_delta_permutation_entropy_all_models.png",
        ylim=None
    )


print("\nFinished process-level pre-collapse entropy summary plots.")