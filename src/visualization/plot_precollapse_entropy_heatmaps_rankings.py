import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)


RESULTS_DIR = "results/2026-06-25_full_benchmark_with_train_predictions"

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "precollapse_entropy",
    "precollapse_entropy_summary.csv"
)

FIGURE_DIR = "figures/precollapse_entropy/heatmaps_rankings"
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
    return name.replace("_", "\n").title()


def plot_heatmap(df, value_col, title, filename, cbar_label):
    heatmap_df = (
        df.pivot(
            index="model",
            columns="series",
            values=value_col
        )
        .reindex(MODEL_ORDER)[SERIES_ORDER]
    )

    values = heatmap_df.values.astype(float)
    plot_values = values + 1e-5

    plt.figure(figsize=(11, 9))

    im = plt.imshow(
        plot_values,
        aspect="auto",
        cmap="magma_r",
        norm=LogNorm(
            vmin=max(1e-5, np.nanmin(plot_values)),
            vmax=np.nanmax(plot_values)
        )
    )

    plt.xticks(
        ticks=np.arange(len(SERIES_ORDER)),
        labels=[series_label(s) for s in SERIES_ORDER],
        fontsize=11
    )

    plt.yticks(
        ticks=np.arange(len(MODEL_ORDER)),
        labels=[model_label(m) for m in MODEL_ORDER],
        fontsize=10
    )

    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label, fontsize=11)

    threshold = np.nanmax(values) / 4

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):

            value = values[i, j]

            text_color = "white" if value > threshold else "black"

            plt.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=text_color
            )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_ranking(df, value_col, title, filename):
    ranking = (
        df.groupby("model")[value_col]
        .mean()
        .reindex(MODEL_ORDER)
        .sort_values()
    )

    plt.figure(figsize=(12, 7))

    bars = plt.barh(
        [model_label(m) for m in ranking.index],
        ranking.values
    )

    plt.gca().invert_yaxis()

    for bar, value in zip(bars, ranking.values):
        plt.text(
            value + ranking.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9
        )

    plt.xlabel("Mean Entropy Error Lower Is Better")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


summary_df = pd.read_csv(SUMMARY_PATH)


# ============================
# Heatmaps
# ============================

plot_heatmap(
    df=summary_df,
    value_col="mean_delta_shannon_entropy",
    title="Pre-Collapse Mean Δ Shannon Entropy",
    filename="precollapse_mean_delta_shannon_heatmap.png",
    cbar_label="Mean Δ Shannon Entropy (log scale)"
)

plot_heatmap(
    df=summary_df,
    value_col="mean_delta_permutation_entropy",
    title="Pre-Collapse Mean Δ Permutation Entropy",
    filename="precollapse_mean_delta_permutation_heatmap.png",
    cbar_label="Mean Δ Permutation Entropy (log scale)"
)


# ============================
# Rankings
# ============================

plot_ranking(
    df=summary_df,
    value_col="mean_delta_shannon_entropy",
    title="Pre-Collapse Shannon Entropy Preservation Ranking",
    filename="precollapse_shannon_entropy_ranking.png"
)

plot_ranking(
    df=summary_df,
    value_col="mean_delta_permutation_entropy",
    title="Pre-Collapse Permutation Entropy Preservation Ranking",
    filename="precollapse_permutation_entropy_ranking.png"
)


# ============================
# Combined ranking
# ============================

summary_df["combined_entropy_error"] = (
    summary_df["mean_delta_shannon_entropy"]
    + summary_df["mean_delta_permutation_entropy"]
)

combined_summary_path = os.path.join(
    RESULTS_DIR,
    "precollapse_entropy",
    "precollapse_entropy_summary_with_combined_score.csv"
)

summary_df.to_csv(combined_summary_path, index=False)

plot_ranking(
    df=summary_df,
    value_col="combined_entropy_error",
    title="Pre-Collapse Overall Entropy Preservation Ranking",
    filename="precollapse_overall_entropy_ranking.png"
)

print("\nFinished heatmaps and rankings.")
print(f"Saved combined summary: {combined_summary_path}")