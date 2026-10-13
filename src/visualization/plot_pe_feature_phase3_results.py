import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm


RESULTS_DIR = "results/pe_transformer_experiment/pe_feature_runs"

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "summaries",
    "pe_feature_transformer_summary.csv"
)

HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")

FIGURE_DIR = "figures/pe_transformer_experiment/phase3_pe_feature"
os.makedirs(FIGURE_DIR, exist_ok=True)

LEARNING_CURVE_DIR = os.path.join(FIGURE_DIR, "learning_curves")
HEATMAP_DIR = os.path.join(FIGURE_DIR, "heatmaps")
RANKING_DIR = os.path.join(FIGURE_DIR, "rankings")
BAR_DIR = os.path.join(FIGURE_DIR, "bar_comparisons")

for d in [LEARNING_CURVE_DIR, HEATMAP_DIR, RANKING_DIR, BAR_DIR]:
    os.makedirs(d, exist_ok=True)


MODEL_ORDER = [
    "simple_causal_transformer",
    "residual_causal_transformer",
    "causal_encoder_transformer"
]

SERIES_ORDER = [
    "linear_gaussian",
    "linear_student_t",
    "nonlinear_gaussian",
    "nonlinear_student_t"
]

EXPERIMENT_ORDER = [
    "sigma_feature",
    "sigma_pe_feature"
]


def model_label(name):
    return name.replace("_", " ").title()


def series_label(name):
    return name.replace("_", "\n").title()


def experiment_label(name):
    if name == "sigma_feature":
        return "Sigma Feature"
    if name == "sigma_pe_feature":
        return "Sigma + PE Feature"
    return name.replace("_", " ").title()


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def plot_learning_curves():
    history_files = glob.glob(os.path.join(HISTORY_DIR, "*_history.csv"))

    all_history = []

    for path in history_files:
        df = pd.read_csv(path)
        all_history.append(df)

        series = df["series"].iloc[0]
        model = df["model"].iloc[0]
        experiment = df["experiment_type"].iloc[0]

        plt.figure(figsize=(9, 5))

        plt.plot(
            df["epoch"],
            df["train_loss"],
            marker="o",
            linewidth=2,
            label="Train loss"
        )

        if "val_loss" in df.columns and df["val_loss"].notna().any():
            plt.plot(
                df["epoch"],
                df["val_loss"],
                marker="o",
                linewidth=2,
                label="Validation loss"
            )

        plt.title(
            f"Learning Curve: {model_label(model)}\n"
            f"{series.replace('_', ' ').title()} | {experiment_label(experiment)}",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()

        out_path = os.path.join(
            LEARNING_CURVE_DIR,
            f"{series}_{model}_{experiment}_learning_curve.png"
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    if all_history:
        combined = pd.concat(all_history, ignore_index=True)

        combined_path = os.path.join(
            RESULTS_DIR,
            "summaries",
            "phase3_training_history_combined.csv"
        )

        combined.to_csv(combined_path, index=False)
        print(f"Saved combined training history: {combined_path}")


def make_pivot(df, value_col, experiment_type):
    subset = df[df["experiment_type"] == experiment_type].copy()

    pivot = (
        subset
        .pivot(index="model", columns="series", values=value_col)
        .reindex(MODEL_ORDER)[SERIES_ORDER]
    )

    return pivot


def plot_metric_heatmap(df, value_col, experiment_type, title, filename, log_scale=False):
    pivot = make_pivot(df, value_col, experiment_type)
    values = pivot.values.astype(float)

    plt.figure(figsize=(9, 5.5))

    if log_scale:
        plot_values = values + 1e-8
        norm = LogNorm(
            vmin=max(np.nanmin(plot_values), 1e-8),
            vmax=np.nanmax(plot_values)
        )
        im = plt.imshow(plot_values, aspect="auto", cmap="magma_r", norm=norm)
    else:
        im = plt.imshow(values, aspect="auto", cmap="magma_r")

    plt.xticks(
        np.arange(len(SERIES_ORDER)),
        [series_label(s) for s in SERIES_ORDER],
        fontsize=10
    )

    plt.yticks(
        np.arange(len(MODEL_ORDER)),
        [model_label(m) for m in MODEL_ORDER],
        fontsize=10
    )

    cbar = plt.colorbar(im)
    cbar.set_label(value_col.replace("_", " ").title(), fontsize=10)

    threshold = np.nanmax(values) / 2

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if value > threshold else "black"

            plt.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color
            )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(HEATMAP_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_improvement_heatmap(df, metric_col, title, filename, higher_is_better=False):
    sigma = make_pivot(df, metric_col, "sigma_feature")
    sigma_pe = make_pivot(df, metric_col, "sigma_pe_feature")

    if higher_is_better:
        improvement = sigma_pe - sigma
        label = "Improvement: Sigma+PE - Sigma"
    else:
        improvement = sigma - sigma_pe
        label = "Improvement: Sigma - Sigma+PE"

    values = improvement.values.astype(float)

    max_abs = np.nanmax(np.abs(values))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    plt.figure(figsize=(9, 5.5))

    im = plt.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        norm=norm
    )

    plt.xticks(
        np.arange(len(SERIES_ORDER)),
        [series_label(s) for s in SERIES_ORDER],
        fontsize=10
    )

    plt.yticks(
        np.arange(len(MODEL_ORDER)),
        [model_label(m) for m in MODEL_ORDER],
        fontsize=10
    )

    cbar = plt.colorbar(im)
    cbar.set_label(label, fontsize=10)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if abs(value) > 0.5 * max_abs else "black"

            plt.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color
            )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(HEATMAP_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    improvement_df = improvement.reset_index()

    csv_path = os.path.join(
        RESULTS_DIR,
        "summaries",
        filename.replace(".png", ".csv")
    )

    improvement_df.to_csv(csv_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved improvement CSV: {csv_path}")


def build_comparison_table(df):
    rows = []

    for series in SERIES_ORDER:
        for model in MODEL_ORDER:

            base = df[
                (df["series"] == series)
                & (df["model"] == model)
                & (df["experiment_type"] == "sigma_feature")
            ]

            pe = df[
                (df["series"] == series)
                & (df["model"] == model)
                & (df["experiment_type"] == "sigma_pe_feature")
            ]

            if base.empty or pe.empty:
                continue

            base = base.iloc[0]
            pe = pe.iloc[0]

            rows.append({
                "series": series,
                "model": model,

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

    comparison_df = pd.DataFrame(rows)

    out_path = os.path.join(
        RESULTS_DIR,
        "summaries",
        "phase3_sigma_vs_sigma_pe_comparison.csv"
    )

    comparison_df.to_csv(out_path, index=False)

    print(f"Saved comparison table: {out_path}")

    return comparison_df


def plot_grouped_bars(comparison_df, series, metric_a, metric_b, ylabel, title, filename):
    subset = comparison_df[comparison_df["series"] == series].copy()

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    values_a = [
        subset[subset["model"] == m][metric_a].iloc[0]
        for m in MODEL_ORDER
    ]

    values_b = [
        subset[subset["model"] == m][metric_b].iloc[0]
        for m in MODEL_ORDER
    ]

    plt.figure(figsize=(10, 5.5))

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
        [model_label(m) for m in MODEL_ORDER],
        rotation=20,
        ha="right"
    )

    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = os.path.join(BAR_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_average_improvement_rankings(comparison_df):
    ranking = (
        comparison_df
        .groupby("model")
        .agg(
            mean_rmse_over_sigma_improvement=(
                "rmse_over_sigma_improvement",
                "mean"
            ),
            mean_directional_accuracy_improvement=(
                "directional_accuracy_improvement",
                "mean"
            )
        )
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    ranking_path = os.path.join(
        RESULTS_DIR,
        "summaries",
        "phase3_average_improvement_ranking.csv"
    )

    ranking.to_csv(ranking_path, index=False)

    print(f"Saved ranking table: {ranking_path}")

    sorted_rmse = ranking.sort_values(
        "mean_rmse_over_sigma_improvement",
        ascending=False
    )

    plt.figure(figsize=(10, 5.5))

    bars = plt.barh(
        [model_label(m) for m in sorted_rmse["model"]],
        sorted_rmse["mean_rmse_over_sigma_improvement"]
    )

    plt.gca().invert_yaxis()

    for bar, value in zip(
        bars,
        sorted_rmse["mean_rmse_over_sigma_improvement"]
    ):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=9
        )

    plt.xlabel("Mean RMSE/σ Improvement: Sigma - Sigma+PE")
    plt.title(
        "Average PE Feature Improvement by Model",
        fontsize=14,
        fontweight="bold"
    )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        RANKING_DIR,
        "phase3_average_rmse_over_sigma_improvement_ranking.png"
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


summary_df = pd.read_csv(SUMMARY_PATH)

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
    if col in summary_df.columns:
        summary_df[col] = safe_numeric(summary_df[col])


# ==========================
# 1. Learning curves
# ==========================

plot_learning_curves()


# ==========================
# 2. Heatmaps for both experiments
# ==========================

for experiment in EXPERIMENT_ORDER:

    plot_metric_heatmap(
        df=summary_df,
        value_col="max_sigma_rmse_over_sigma",
        experiment_type=experiment,
        title=f"Phase 3: Max-Sigma RMSE/σ — {experiment_label(experiment)}",
        filename=f"phase3_{experiment}_max_sigma_rmse_over_sigma_heatmap.png",
        log_scale=True
    )

    plot_metric_heatmap(
        df=summary_df,
        value_col="max_sigma_directional_accuracy",
        experiment_type=experiment,
        title=f"Phase 3: Max-Sigma Directional Accuracy — {experiment_label(experiment)}",
        filename=f"phase3_{experiment}_max_sigma_directional_accuracy_heatmap.png",
        log_scale=False
    )


# ==========================
# 3. Improvement heatmaps
# ==========================

plot_improvement_heatmap(
    df=summary_df,
    metric_col="max_sigma_rmse_over_sigma",
    title="Phase 3: PE Feature Improvement in RMSE/σ",
    filename="phase3_rmse_over_sigma_improvement_heatmap.png",
    higher_is_better=False
)

plot_improvement_heatmap(
    df=summary_df,
    metric_col="max_sigma_directional_accuracy",
    title="Phase 3: PE Feature Improvement in Directional Accuracy",
    filename="phase3_directional_accuracy_improvement_heatmap.png",
    higher_is_better=True
)


# ==========================
# 4. Comparison table + grouped bars
# ==========================

comparison_df = build_comparison_table(summary_df)

for series in SERIES_ORDER:

    plot_grouped_bars(
        comparison_df=comparison_df,
        series=series,
        metric_a="sigma_rmse_over_sigma",
        metric_b="sigma_pe_rmse_over_sigma",
        ylabel="Max-Sigma RMSE/σ",
        title=f"Phase 3 RMSE/σ Comparison: {series.replace('_', ' ').title()}",
        filename=f"phase3_{series}_rmse_over_sigma_grouped_bar.png"
    )

    plot_grouped_bars(
        comparison_df=comparison_df,
        series=series,
        metric_a="sigma_directional_accuracy",
        metric_b="sigma_pe_directional_accuracy",
        ylabel="Max-Sigma Directional Accuracy",
        title=f"Phase 3 Directional Accuracy Comparison: {series.replace('_', ' ').title()}",
        filename=f"phase3_{series}_directional_accuracy_grouped_bar.png"
    )


# ==========================
# 5. Average improvement ranking
# ==========================

plot_average_improvement_rankings(comparison_df)

print("\nFinished Phase 3 PE-feature plotting.")
print(f"Figures saved in: {FIGURE_DIR}")