import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results/2026-06-25_full_benchmark"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

CORRECTED_SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")
FIGURE_DIR = "figures/corrected_benchmark_v1"

os.makedirs(CORRECTED_SUMMARY_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

COLLAPSE_THRESHOLD = 1.0

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


def load_metrics():
    files = glob.glob(os.path.join(METRICS_DIR, "*_metrics.csv"))
    dfs = []

    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def find_first_collapse_row(df):
    df = df.sort_values("sigma").reset_index(drop=True)
    collapsed = df[df["rmse_over_sigma"] >= COLLAPSE_THRESHOLD]

    if collapsed.empty:
        return None

    return collapsed.iloc[0]


metrics_df = load_metrics()

summary_rows = []

for (series, model), group in metrics_df.groupby(["series", "model"]):

    group = group.sort_values("sigma").reset_index(drop=True)

    last_row = group.iloc[-1]
    collapse_row = find_first_collapse_row(group)

    if collapse_row is None:
        collapse_status = "not confirmed"
        collapse_sigma = "> max_sigma"

        # If no collapse, use final available sigma as reference.
        collapse_rmse = last_row["rmse"]
        collapse_rmse_over_sigma = last_row["rmse_over_sigma"]
        collapse_directional_accuracy = last_row["directional_accuracy"]

        pre_collapse_df = group.copy()

    else:
        collapse_status = "confirmed"
        collapse_sigma = collapse_row["sigma"]

        collapse_rmse = collapse_row["rmse"]
        collapse_rmse_over_sigma = collapse_row["rmse_over_sigma"]
        collapse_directional_accuracy = collapse_row["directional_accuracy"]

        pre_collapse_df = group[group["sigma"] <= collapse_sigma].copy()

    summary_rows.append({
        "series": series,
        "model": model,

        "collapse_status": collapse_status,
        "collapse_sigma": collapse_sigma,

        "collapse_rmse": collapse_rmse,
        "collapse_rmse_over_sigma": collapse_rmse_over_sigma,
        "collapse_directional_accuracy": collapse_directional_accuracy,

        "precollapse_mean_rmse": pre_collapse_df["rmse"].mean(),
        "precollapse_mean_rmse_over_sigma": pre_collapse_df["rmse_over_sigma"].mean(),
        "precollapse_mean_directional_accuracy": pre_collapse_df["directional_accuracy"].mean(),

        "max_sigma": last_row["sigma"],
        "max_sigma_rmse": last_row["rmse"],
        "max_sigma_rmse_over_sigma": last_row["rmse_over_sigma"],
        "max_sigma_directional_accuracy": last_row["directional_accuracy"]
    })


corrected_summary = pd.DataFrame(summary_rows)

corrected_summary_path = os.path.join(
    CORRECTED_SUMMARY_DIR,
    "corrected_benchmark_v1_summary.csv"
)

corrected_summary.to_csv(corrected_summary_path, index=False)

print(f"Saved corrected summary: {corrected_summary_path}")


def plot_heatmap(value_column, title, file_name, cap_value=None):

    heatmap_df = corrected_summary.pivot(
        index="model",
        columns="series",
        values=value_column
    ).reindex(MODEL_ORDER)[SERIES_ORDER]

    if cap_value is not None:
        plot_values = heatmap_df.clip(upper=cap_value)
    else:
        plot_values = heatmap_df.copy()

    plt.figure(figsize=(12, 9))

    im = plt.imshow(
        plot_values.values,
        aspect="auto",
        cmap="magma_r"
    )

    plt.xticks(
        ticks=np.arange(len(SERIES_ORDER)),
        labels=[s.replace("_", "\n") for s in SERIES_ORDER],
        fontsize=11
    )

    plt.yticks(
        ticks=np.arange(len(MODEL_ORDER)),
        labels=[model_label(m) for m in MODEL_ORDER],
        fontsize=10
    )

    cbar = plt.colorbar(im)
    cbar.set_label(value_column.replace("_", " ").title(), fontsize=11)

    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            raw_value = heatmap_df.values[i, j]
            plotted_value = plot_values.values[i, j]

            if cap_value is not None and raw_value > cap_value:
                text = f">{cap_value:.1f}"
            else:
                text = f"{raw_value:.2f}"

            text_color = "white" if plotted_value > np.nanmax(plot_values.values) * 0.55 else "black"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color
            )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, file_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# 1. Correct collapse-point RMSE/sigma heatmap
plot_heatmap(
    value_column="collapse_rmse_over_sigma",
    title="Corrected Collapse-Point RMSE/Sigma",
    file_name="corrected_collapse_point_rmse_over_sigma_heatmap.png",
    cap_value=5.0
)

# 2. Pre-collapse mean RMSE/sigma heatmap
plot_heatmap(
    value_column="precollapse_mean_rmse_over_sigma",
    title="Corrected Pre-Collapse Mean RMSE/Sigma",
    file_name="corrected_precollapse_mean_rmse_over_sigma_heatmap.png",
    cap_value=5.0
)

# 3. Max-sigma RMSE/sigma heatmap, explicitly named
plot_heatmap(
    value_column="max_sigma_rmse_over_sigma",
    title="Maximum-Tested-Sigma RMSE/Sigma",
    file_name="corrected_max_sigma_rmse_over_sigma_heatmap.png",
    cap_value=5.0
)


# ============================
# Corrected average ranks
# ============================

for value_column, file_name, title in [
    (
        "collapse_rmse_over_sigma",
        "corrected_average_rank_by_collapse_rmse_over_sigma.png",
        "Average Rank by Collapse-Point RMSE/Sigma"
    ),
    (
        "precollapse_mean_rmse_over_sigma",
        "corrected_average_rank_by_precollapse_mean_rmse_over_sigma.png",
        "Average Rank by Pre-Collapse Mean RMSE/Sigma"
    ),
    (
        "max_sigma_rmse_over_sigma",
        "corrected_average_rank_by_max_sigma_rmse_over_sigma.png",
        "Average Rank by Maximum-Sigma RMSE/Sigma"
    )
]:

    rank_df = corrected_summary.copy()

    rank_df["rank"] = rank_df.groupby("series")[value_column].rank(
        method="average",
        ascending=True
    )

    avg_rank = (
        rank_df.groupby("model")["rank"]
        .mean()
        .reindex(MODEL_ORDER)
        .sort_values()
    )

    plt.figure(figsize=(12, 7))

    bars = plt.bar(
        [model_label(m) for m in avg_rank.index],
        avg_rank.values
    )

    for bar, value in zip(bars, avg_rank.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{value:.2f}",
            ha="center",
            fontsize=9
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Average Rank Lower Is Better")
    plt.xticks(rotation=60, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, file_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


print("\nCorrected benchmark plots completed.")