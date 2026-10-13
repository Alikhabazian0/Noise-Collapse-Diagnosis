import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results/pe_transformer_experiment/pe_feature_runs"
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")

COMPARISON_PATH = os.path.join(
    SUMMARY_DIR,
    "phase3_sigma_vs_sigma_pe_comparison.csv"
)

OUTPUT_DIR = "figures/pe_transformer_experiment/phase3_pe_feature/final_summary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_DIR = os.path.join(SUMMARY_DIR, "final_summary_tables")
os.makedirs(CSV_DIR, exist_ok=True)


def model_label(name):
    return name.replace("_", " ").title()


def series_label(name):
    return name.replace("_", " ").title().replace("Student T", "Student-t")


def conclusion_from_gains(da_gain, rmse_gain):
    if da_gain >= 0.03 and rmse_gain > 0:
        return "Strong PE benefit"
    if da_gain >= 0.01 and rmse_gain > 0:
        return "Moderate PE benefit"
    if da_gain > 0 and rmse_gain > 0:
        return "Small PE benefit"
    if da_gain <= 0 and rmse_gain <= 0:
        return "PE worsened both metrics"
    if da_gain <= 0 and rmse_gain > 0:
        return "RMSE benefit only"
    if da_gain > 0 and rmse_gain <= 0:
        return "Directional benefit only"
    return "Mixed effect"


def save_grouped_bar(
    df,
    category_col,
    rmse_col,
    da_col,
    title,
    filename,
    xlabel
):
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        df[rmse_col],
        width,
        label="Avg RMSE/σ improvement"
    )

    plt.bar(
        x + width / 2,
        df[da_col],
        width,
        label="Avg directional accuracy improvement"
    )

    plt.axhline(0, linewidth=1)

    plt.xticks(
        x,
        df[category_col],
        rotation=25,
        ha="right"
    )

    plt.ylabel("Average improvement")
    plt.xlabel(xlabel)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {path}")


def save_horizontal_bar(
    df,
    category_col,
    value_col,
    title,
    filename,
    xlabel
):
    plot_df = df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(11, 6))

    bars = plt.barh(
        plot_df[category_col],
        plot_df[value_col]
    )

    plt.axvline(0, linewidth=1)

    for bar, value in zip(bars, plot_df[value_col]):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=9
        )

    plt.xlabel(xlabel)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {path}")


def save_table_plot(df, title, filename):
    display_df = df.copy()

    for col in ["da_gain", "rmse_gain"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:+.3f}")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")

    plt.title(title, fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {path}")


df = pd.read_csv(COMPARISON_PATH)

required_cols = [
    "series",
    "model",
    "rmse_over_sigma_improvement",
    "directional_accuracy_improvement"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in comparison file: {missing}")

df["series_label"] = df["series"].apply(series_label)
df["model_label"] = df["model"].apply(model_label)


# ============================================================
# 1. Average RMSE improvement and directional improvement by model
# ============================================================

by_model = (
    df.groupby(["model", "model_label"])
    .agg(
        avg_rmse_improvement=("rmse_over_sigma_improvement", "mean"),
        avg_direction_improvement=("directional_accuracy_improvement", "mean")
    )
    .reset_index()
    .sort_values("avg_rmse_improvement", ascending=False)
)

by_model_csv = os.path.join(
    CSV_DIR,
    "phase3_avg_improvement_by_model.csv"
)
by_model.to_csv(by_model_csv, index=False)

save_grouped_bar(
    df=by_model,
    category_col="model_label",
    rmse_col="avg_rmse_improvement",
    da_col="avg_direction_improvement",
    title="Phase 3: Average PE Improvement by Model",
    filename="phase3_avg_improvement_by_model.png",
    xlabel="Model"
)


# ============================================================
# 2. Average RMSE improvement and directional improvement by time series
# ============================================================

by_series = (
    df.groupby(["series", "series_label"])
    .agg(
        avg_rmse_improvement=("rmse_over_sigma_improvement", "mean"),
        avg_direction_improvement=("directional_accuracy_improvement", "mean")
    )
    .reset_index()
    .sort_values("avg_rmse_improvement", ascending=False)
)

by_series_csv = os.path.join(
    CSV_DIR,
    "phase3_avg_improvement_by_time_series.csv"
)
by_series.to_csv(by_series_csv, index=False)

save_grouped_bar(
    df=by_series,
    category_col="series_label",
    rmse_col="avg_rmse_improvement",
    da_col="avg_direction_improvement",
    title="Phase 3: Average PE Improvement by Time Series",
    filename="phase3_avg_improvement_by_time_series.png",
    xlabel="Time series"
)


# ============================================================
# 3. Mean improvement based on time series
#    Composite score uses standardized RMSE and DA gains.
# ============================================================

series_scores = by_series.copy()

rmse_std = series_scores["avg_rmse_improvement"].std(ddof=0)
da_std = series_scores["avg_direction_improvement"].std(ddof=0)

series_scores["rmse_z"] = (
    (series_scores["avg_rmse_improvement"]
     - series_scores["avg_rmse_improvement"].mean())
    / (rmse_std + 1e-12)
)

series_scores["direction_z"] = (
    (series_scores["avg_direction_improvement"]
     - series_scores["avg_direction_improvement"].mean())
    / (da_std + 1e-12)
)

series_scores["mean_standardized_improvement"] = (
    series_scores["rmse_z"] + series_scores["direction_z"]
) / 2

mean_series_csv = os.path.join(
    CSV_DIR,
    "phase3_mean_standardized_improvement_by_time_series.csv"
)
series_scores.to_csv(mean_series_csv, index=False)

save_horizontal_bar(
    df=series_scores,
    category_col="series_label",
    value_col="mean_standardized_improvement",
    title="Phase 3: Mean Standardized PE Improvement by Time Series",
    filename="phase3_mean_standardized_improvement_by_time_series.png",
    xlabel="Mean standardized improvement"
)


# ============================================================
# 4. Final all-in-one summary table
#    Best model selected mainly by directional accuracy gain.
# ============================================================

summary_rows = []

for series, group in df.groupby("series"):
    group = group.copy()

    # Primary criterion: directional accuracy gain.
    # Secondary criterion: RMSE/σ gain.
    best_row = group.sort_values(
        ["directional_accuracy_improvement", "rmse_over_sigma_improvement"],
        ascending=False
    ).iloc[0]

    da_gain = best_row["directional_accuracy_improvement"]
    rmse_gain = best_row["rmse_over_sigma_improvement"]

    summary_rows.append({
        "stochastic_process": series_label(series),
        "best_model": model_label(best_row["model"]),
        "da_gain": da_gain,
        "rmse_gain": rmse_gain,
        "conclusion": conclusion_from_gains(da_gain, rmse_gain)
    })

final_summary = pd.DataFrame(summary_rows)

process_order = [
    "Linear Gaussian",
    "Linear Student-t",
    "Nonlinear Gaussian",
    "Nonlinear Student-t"
]

final_summary["order"] = final_summary["stochastic_process"].apply(
    lambda x: process_order.index(x) if x in process_order else 999
)

final_summary = (
    final_summary
    .sort_values("order")
    .drop(columns=["order"])
    .reset_index(drop=True)
)

final_summary_csv = os.path.join(
    CSV_DIR,
    "phase3_final_interpretive_summary.csv"
)
final_summary.to_csv(final_summary_csv, index=False)

save_table_plot(
    df=final_summary,
    title="Phase 3: Best PE Feature Effect by Stochastic Process",
    filename="phase3_final_interpretive_summary_table.png"
)


print("\nFinished Phase 3 final summary tables and plots.")
print(f"CSV files saved in: {CSV_DIR}")
print(f"Plots saved in: {OUTPUT_DIR}")

print("\nGenerated CSVs:")
print(by_model_csv)
print(by_series_csv)
print(mean_series_csv)
print(final_summary_csv)