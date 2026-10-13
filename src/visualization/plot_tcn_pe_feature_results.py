import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


RESULTS_DIR = "results/pe_transformer_experiment/tcn_pe_feature_runs"

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "summaries",
    "tcn_pe_feature_summary.csv"
)

HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")

FIGURE_DIR = "figures/pe_transformer_experiment/tcn_pe_feature"
LEARNING_CURVE_DIR = os.path.join(FIGURE_DIR, "learning_curves")
HEATMAP_DIR = os.path.join(FIGURE_DIR, "heatmaps")
BAR_DIR = os.path.join(FIGURE_DIR, "bar_comparisons")
FINAL_DIR = os.path.join(FIGURE_DIR, "final_summary")
RANKING_DIR = os.path.join(FIGURE_DIR, "rankings")

for d in [
    FIGURE_DIR,
    LEARNING_CURVE_DIR,
    HEATMAP_DIR,
    BAR_DIR,
    FINAL_DIR,
    RANKING_DIR,
]:
    os.makedirs(d, exist_ok=True)

CSV_DIR = os.path.join(
    RESULTS_DIR,
    "summaries",
    "tcn_summary_tables"
)
os.makedirs(CSV_DIR, exist_ok=True)


SERIES_ORDER = [
    "linear_gaussian",
    "linear_student_t",
    "nonlinear_gaussian",
    "nonlinear_student_t",
]

EXPERIMENT_ORDER = [
    "sigma_feature",
    "sigma_pe_feature",
]


def series_label(name):
    return name.replace("_", " ").title().replace("Student T", "Student-t")


def experiment_label(name):
    if name == "sigma_feature":
        return "Sigma Feature"
    if name == "sigma_pe_feature":
        return "Sigma + PE Feature"
    return name.replace("_", " ").title()


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


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


def plot_learning_curves():
    files = glob.glob(os.path.join(HISTORY_DIR, "*_history.csv"))
    all_history = []

    for path in files:
        df = pd.read_csv(path)
        all_history.append(df)

        series = df["series"].iloc[0]
        experiment = df["experiment_type"].iloc[0]

        plt.figure(figsize=(9, 5.5))

        plt.plot(
            df["epoch"],
            df["train_loss"],
            marker="o",
            linewidth=2,
            label="Train loss"
        )

        if "val_loss" in df.columns:
            plt.plot(
                df["epoch"],
                df["val_loss"],
                marker="o",
                linewidth=2,
                label="Validation loss"
            )

        plt.title(
            f"TCN Learning Curve\n{series_label(series)} | {experiment_label(experiment)}",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            LEARNING_CURVE_DIR,
            f"{series}_tcn_level_{experiment}_learning_curve.png"
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    if all_history:
        combined = pd.concat(all_history, ignore_index=True)

        out_path = os.path.join(
            CSV_DIR,
            "tcn_training_history_combined.csv"
        )

        combined.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


def build_comparison_table(summary):
    rows = []

    for series in SERIES_ORDER:
        base = summary[
            (summary["series"] == series)
            & (summary["experiment_type"] == "sigma_feature")
        ]

        pe = summary[
            (summary["series"] == series)
            & (summary["experiment_type"] == "sigma_pe_feature")
        ]

        if base.empty or pe.empty:
            continue

        base = base.iloc[0]
        pe = pe.iloc[0]

        rows.append({
            "series": series,
            "model": "tcn_level",

            "sigma_rmse_over_sigma": base["max_sigma_rmse_over_sigma"],
            "sigma_pe_rmse_over_sigma": pe["max_sigma_rmse_over_sigma"],
            "rmse_over_sigma_improvement": (
                base["max_sigma_rmse_over_sigma"]
                - pe["max_sigma_rmse_over_sigma"]
            ),

            "sigma_directional_accuracy": base["max_sigma_directional_accuracy"],
            "sigma_pe_directional_accuracy": pe["max_sigma_directional_accuracy"],
            "directional_accuracy_improvement": (
                pe["max_sigma_directional_accuracy"]
                - base["max_sigma_directional_accuracy"]
            ),

            "sigma_collapse_status": base["collapse_status"],
            "sigma_pe_collapse_status": pe["collapse_status"],
            "sigma_collapse_sigma": base["collapse_sigma"],
            "sigma_pe_collapse_sigma": pe["collapse_sigma"],
        })

    comparison = pd.DataFrame(rows)

    out_path = os.path.join(
        CSV_DIR,
        "tcn_sigma_vs_sigma_pe_comparison.csv"
    )

    comparison.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return comparison


def plot_grouped_metric_bars(comparison, metric_a, metric_b, ylabel, title, filename):
    x = np.arange(len(SERIES_ORDER))
    width = 0.35

    values_a = []
    values_b = []

    for series in SERIES_ORDER:
        row = comparison[comparison["series"] == series].iloc[0]
        values_a.append(row[metric_a])
        values_b.append(row[metric_b])

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - width / 2,
        values_a,
        width,
        label="Sigma Feature"
    )

    plt.bar(
        x + width / 2,
        values_b,
        width,
        label="Sigma + PE Feature"
    )

    plt.xticks(
        x,
        [series_label(s) for s in SERIES_ORDER],
        rotation=25,
        ha="right"
    )

    plt.ylabel(ylabel)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(BAR_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_improvement_bar(comparison, value_col, title, filename, xlabel):
    plot_df = comparison.copy()
    plot_df["series_label"] = plot_df["series"].apply(series_label)
    plot_df = plot_df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(11, 6))

    bars = plt.barh(
        plot_df["series_label"],
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

    out_path = os.path.join(RANKING_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_improvement_heatmap(comparison, metric_col, title, filename):
    values = comparison.set_index("series").reindex(SERIES_ORDER)[metric_col].values
    values = values.reshape(1, -1).astype(float)

    max_abs = np.nanmax(np.abs(values))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    plt.figure(figsize=(10, 2.8))

    im = plt.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        norm=norm
    )

    plt.xticks(
        np.arange(len(SERIES_ORDER)),
        [series_label(s).replace(" ", "\n") for s in SERIES_ORDER],
        fontsize=10
    )

    plt.yticks([0], ["TCN"], fontsize=11)

    cbar = plt.colorbar(im)
    cbar.set_label("Improvement")

    for j in range(values.shape[1]):
        value = values[0, j]
        color = "white" if abs(value) > 0.5 * max_abs else "black"

        plt.text(
            j,
            0,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(HEATMAP_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(
        CSV_DIR,
        filename.replace(".png", ".csv")
    )

    pd.DataFrame({
        "series": SERIES_ORDER,
        metric_col: values.flatten()
    }).to_csv(csv_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {csv_path}")


def save_table_plot(df, title, filename):
    display = df.copy()

    for col in ["da_gain", "rmse_gain"]:
        display[col] = display[col].map(lambda x: f"{x:+.3f}")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
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

    out_path = os.path.join(FINAL_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


summary = pd.read_csv(SUMMARY_PATH)

numeric_cols = [
    "collapse_rmse",
    "collapse_rmse_over_sigma",
    "collapse_directional_accuracy",
    "precollapse_mean_rmse",
    "precollapse_mean_rmse_over_sigma",
    "precollapse_mean_directional_accuracy",
    "max_sigma",
    "max_sigma_rmse",
    "max_sigma_rmse_over_sigma",
    "max_sigma_directional_accuracy",
]

for col in numeric_cols:
    if col in summary.columns:
        summary[col] = safe_numeric(summary[col])


# 1. Learning curves
plot_learning_curves()


# 2. Comparison table
comparison = build_comparison_table(summary)


# 3. Grouped bar comparisons
plot_grouped_metric_bars(
    comparison=comparison,
    metric_a="sigma_rmse_over_sigma",
    metric_b="sigma_pe_rmse_over_sigma",
    ylabel="Max-Sigma RMSE/σ",
    title="TCN PE Feature Comparison: RMSE/σ",
    filename="tcn_rmse_over_sigma_grouped_bar.png"
)

plot_grouped_metric_bars(
    comparison=comparison,
    metric_a="sigma_directional_accuracy",
    metric_b="sigma_pe_directional_accuracy",
    ylabel="Max-Sigma Directional Accuracy",
    title="TCN PE Feature Comparison: Directional Accuracy",
    filename="tcn_directional_accuracy_grouped_bar.png"
)


# 4. Improvement heatmaps
plot_improvement_heatmap(
    comparison=comparison,
    metric_col="rmse_over_sigma_improvement",
    title="TCN PE Feature Improvement in RMSE/σ",
    filename="tcn_rmse_over_sigma_improvement_heatmap.png"
)

plot_improvement_heatmap(
    comparison=comparison,
    metric_col="directional_accuracy_improvement",
    title="TCN PE Feature Improvement in Directional Accuracy",
    filename="tcn_directional_accuracy_improvement_heatmap.png"
)


# 5. Rankings
plot_improvement_bar(
    comparison=comparison,
    value_col="rmse_over_sigma_improvement",
    title="TCN RMSE/σ Improvement Ranking",
    filename="tcn_rmse_over_sigma_improvement_ranking.png",
    xlabel="RMSE/σ improvement: Sigma - Sigma+PE"
)

plot_improvement_bar(
    comparison=comparison,
    value_col="directional_accuracy_improvement",
    title="TCN Directional Accuracy Improvement Ranking",
    filename="tcn_directional_accuracy_improvement_ranking.png",
    xlabel="Directional accuracy improvement: Sigma+PE - Sigma"
)


# 6. Average summary
avg_summary = pd.DataFrame([{
    "model": "tcn_level",
    "avg_rmse_improvement": comparison["rmse_over_sigma_improvement"].mean(),
    "avg_directional_accuracy_improvement": comparison["directional_accuracy_improvement"].mean(),
    "positive_rmse_cases": int((comparison["rmse_over_sigma_improvement"] > 0).sum()),
    "positive_direction_cases": int((comparison["directional_accuracy_improvement"] > 0).sum()),
    "total_cases": len(comparison),
}])

avg_summary_path = os.path.join(
    CSV_DIR,
    "tcn_average_improvement_summary.csv"
)

avg_summary.to_csv(avg_summary_path, index=False)
print(f"Saved: {avg_summary_path}")


# 7. Final interpretive table
rows = []

for _, row in comparison.iterrows():
    da_gain = row["directional_accuracy_improvement"]
    rmse_gain = row["rmse_over_sigma_improvement"]

    rows.append({
        "stochastic_process": series_label(row["series"]),
        "model": "TCN",
        "da_gain": da_gain,
        "rmse_gain": rmse_gain,
        "conclusion": conclusion_from_gains(da_gain, rmse_gain)
    })

final_summary = pd.DataFrame(rows)

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

final_summary_path = os.path.join(
    CSV_DIR,
    "tcn_final_interpretive_summary.csv"
)

final_summary.to_csv(final_summary_path, index=False)
print(f"Saved: {final_summary_path}")

save_table_plot(
    df=final_summary,
    title="TCN PE Feature Effect by Stochastic Process",
    filename="tcn_final_interpretive_summary_table.png"
)


print("\nFinished TCN PE-feature plotting.")
print(f"Figures saved in: {FIGURE_DIR}")
print(f"CSV tables saved in: {CSV_DIR}")