import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm


RESULTS_DIR = "results/pe_transformer_experiment/pe_attention_runs"

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "summaries",
    "pe_attention_transformer_summary.csv"
)

HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")

FIGURE_DIR = "figures/pe_transformer_experiment/phase5_pe_attention"

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
    "phase5_summary_tables"
)
os.makedirs(CSV_DIR, exist_ok=True)


MODEL_ORDER = [
    "simple_causal_transformer",
    "residual_causal_transformer",
    "causal_encoder_transformer",
]

SERIES_ORDER = [
    "linear_gaussian",
    "linear_student_t",
    "nonlinear_gaussian",
    "nonlinear_student_t",
]

EXPERIMENT_ORDER = [
    "standard_attention",
    "pe_biased_attention",
]


def model_label(name):
    return name.replace("_", " ").title()


def series_label(name):
    return name.replace("_", " ").title().replace("Student T", "Student-t")


def experiment_label(name):
    if name == "standard_attention":
        return "Standard Attention"
    if name == "pe_biased_attention":
        return "PE-Biased Attention"
    return name.replace("_", " ").title()


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def make_pivot(df, value_col, experiment_type):
    subset = df[df["experiment_type"] == experiment_type].copy()

    return (
        subset
        .pivot(index="model", columns="series", values=value_col)
        .reindex(MODEL_ORDER)[SERIES_ORDER]
    )


def plot_learning_curves():
    files = glob.glob(os.path.join(HISTORY_DIR, "*_history.csv"))

    all_history = []

    for path in files:
        df = pd.read_csv(path)
        all_history.append(df)

        series = df["series"].iloc[0]
        model = df["model"].iloc[0]
        experiment = df["experiment_type"].iloc[0]

        plt.figure(figsize=(10, 6))

        plt.plot(
            df["epoch"],
            df["train_loss"],
            marker="o",
            linewidth=2,
            label="Train loss",
        )

        if "val_loss" in df.columns:
            plt.plot(
                df["epoch"],
                df["val_loss"],
                marker="o",
                linewidth=2,
                label="Validation loss",
            )

        plt.title(
            f"Phase 5 Learning Curve: {model_label(model)}\n"
            f"{series_label(series)} | {experiment_label(experiment)}",
            fontsize=13,
            fontweight="bold",
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            LEARNING_CURVE_DIR,
            f"{series}_{model}_{experiment}_learning_curve.png",
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    if all_history:
        combined = pd.concat(all_history, ignore_index=True)

        out_path = os.path.join(
            CSV_DIR,
            "phase5_training_history_combined.csv"
        )

        combined.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


def plot_metric_heatmap(df, value_col, experiment_type, title, filename, log_scale=False):
    pivot = make_pivot(df, value_col, experiment_type)

    values = pivot.values.astype(float)

    plt.figure(figsize=(9, 5.5))

    if log_scale:
        plot_values = values + 1e-8
        norm = LogNorm(
            vmin=max(np.nanmin(plot_values), 1e-8),
            vmax=np.nanmax(plot_values),
        )

        im = plt.imshow(
            plot_values,
            aspect="auto",
            cmap="magma_r",
            norm=norm,
        )
    else:
        im = plt.imshow(
            values,
            aspect="auto",
            cmap="magma_r",
        )

    plt.xticks(
        np.arange(len(SERIES_ORDER)),
        [series_label(s).replace(" ", "\n") for s in SERIES_ORDER],
        fontsize=10,
    )

    plt.yticks(
        np.arange(len(MODEL_ORDER)),
        [model_label(m) for m in MODEL_ORDER],
        fontsize=10,
    )

    cbar = plt.colorbar(im)
    cbar.set_label(value_col.replace("_", " ").title())

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
                color=color,
            )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(HEATMAP_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_improvement_heatmap(df, metric_col, title, filename, higher_is_better=False):
    base = make_pivot(df, metric_col, "standard_attention")
    pe = make_pivot(df, metric_col, "pe_biased_attention")

    if higher_is_better:
        improvement = pe - base
        label = "Improvement: PE Attention - Standard"
    else:
        improvement = base - pe
        label = "Improvement: Standard - PE Attention"

    values = improvement.values.astype(float)

    max_abs = np.nanmax(np.abs(values))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    plt.figure(figsize=(9, 5.5))

    im = plt.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
    )

    plt.xticks(
        np.arange(len(SERIES_ORDER)),
        [series_label(s).replace(" ", "\n") for s in SERIES_ORDER],
        fontsize=10,
    )

    plt.yticks(
        np.arange(len(MODEL_ORDER)),
        [model_label(m) for m in MODEL_ORDER],
        fontsize=10,
    )

    cbar = plt.colorbar(im)
    cbar.set_label(label)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if abs(value) > 0.5 * max_abs else "black"

            plt.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(HEATMAP_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(
        CSV_DIR,
        filename.replace(".png", ".csv"),
    )

    improvement.reset_index().to_csv(csv_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {csv_path}")


def build_comparison_table(df):
    rows = []

    for series in SERIES_ORDER:
        for model in MODEL_ORDER:

            base = df[
                (df["series"] == series)
                & (df["model"] == model)
                & (df["experiment_type"] == "standard_attention")
            ]

            pe = df[
                (df["series"] == series)
                & (df["model"] == model)
                & (df["experiment_type"] == "pe_biased_attention")
            ]

            if base.empty or pe.empty:
                continue

            base = base.iloc[0]
            pe = pe.iloc[0]

            rows.append({
                "series": series,
                "model": model,

                "standard_rmse_over_sigma": base["max_sigma_rmse_over_sigma"],
                "pe_attention_rmse_over_sigma": pe["max_sigma_rmse_over_sigma"],
                "rmse_over_sigma_improvement": (
                    base["max_sigma_rmse_over_sigma"]
                    - pe["max_sigma_rmse_over_sigma"]
                ),

                "standard_directional_accuracy": base["max_sigma_directional_accuracy"],
                "pe_attention_directional_accuracy": pe["max_sigma_directional_accuracy"],
                "directional_accuracy_improvement": (
                    pe["max_sigma_directional_accuracy"]
                    - base["max_sigma_directional_accuracy"]
                ),

                "standard_collapse_status": base["collapse_status"],
                "pe_attention_collapse_status": pe["collapse_status"],
                "standard_collapse_sigma": base["collapse_sigma"],
                "pe_attention_collapse_sigma": pe["collapse_sigma"],
            })

    comparison = pd.DataFrame(rows)

    out_path = os.path.join(
        CSV_DIR,
        "phase5_standard_vs_pe_attention_comparison.csv",
    )

    comparison.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return comparison


def plot_grouped_bars(comparison, series, metric_a, metric_b, ylabel, title, filename):
    subset = comparison[comparison["series"] == series].copy()

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
        label="Standard Attention",
    )

    plt.bar(
        x + width / 2,
        values_b,
        width,
        label="PE-Biased Attention",
    )

    plt.xticks(
        x,
        [model_label(m) for m in MODEL_ORDER],
        rotation=20,
        ha="right",
    )

    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(BAR_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def save_grouped_summary_bar(df, category_col, rmse_col, da_col, title, filename, xlabel):
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        df[rmse_col],
        width,
        label="Avg RMSE/σ improvement",
    )

    plt.bar(
        x + width / 2,
        df[da_col],
        width,
        label="Avg directional accuracy improvement",
    )

    plt.axhline(0, linewidth=1)

    plt.xticks(
        x,
        df[category_col],
        rotation=25,
        ha="right",
    )

    plt.ylabel("Average improvement")
    plt.xlabel(xlabel)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FINAL_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def save_horizontal_bar(df, category_col, value_col, title, filename, xlabel, output_dir):
    plot_df = df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(11, 6))

    bars = plt.barh(
        plot_df[category_col],
        plot_df[value_col],
    )

    plt.axvline(0, linewidth=1)

    for bar, value in zip(bars, plot_df[value_col]):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=9,
        )

    plt.xlabel(xlabel)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def conclusion_from_gains(da_gain, rmse_gain):
    if da_gain >= 0.03 and rmse_gain > 0:
        return "Strong PE-attention benefit"
    if da_gain >= 0.01 and rmse_gain > 0:
        return "Moderate PE-attention benefit"
    if da_gain > 0 and rmse_gain > 0:
        return "Small PE-attention benefit"
    if da_gain <= 0 and rmse_gain <= 0:
        return "PE attention worsened both metrics"
    if da_gain <= 0 and rmse_gain > 0:
        return "RMSE benefit only"
    if da_gain > 0 and rmse_gain <= 0:
        return "Directional benefit only"
    return "Mixed effect"


def save_table_plot(df, title, filename):
    display = df.copy()

    for col in ["da_gain", "rmse_gain"]:
        display[col] = display[col].map(lambda x: f"{x:+.3f}")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
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


# 2. Main metric heatmaps
for experiment in EXPERIMENT_ORDER:

    plot_metric_heatmap(
        df=summary,
        value_col="max_sigma_rmse_over_sigma",
        experiment_type=experiment,
        title=f"Phase 5: Max-Sigma RMSE/σ — {experiment_label(experiment)}",
        filename=f"phase5_{experiment}_max_sigma_rmse_over_sigma_heatmap.png",
        log_scale=True,
    )

    plot_metric_heatmap(
        df=summary,
        value_col="max_sigma_directional_accuracy",
        experiment_type=experiment,
        title=f"Phase 5: Max-Sigma Directional Accuracy — {experiment_label(experiment)}",
        filename=f"phase5_{experiment}_max_sigma_directional_accuracy_heatmap.png",
        log_scale=False,
    )


# 3. Improvement heatmaps
plot_improvement_heatmap(
    df=summary,
    metric_col="max_sigma_rmse_over_sigma",
    title="Phase 5: PE-Attention Improvement in RMSE/σ",
    filename="phase5_rmse_over_sigma_improvement_heatmap.png",
    higher_is_better=False,
)

plot_improvement_heatmap(
    df=summary,
    metric_col="max_sigma_directional_accuracy",
    title="Phase 5: PE-Attention Improvement in Directional Accuracy",
    filename="phase5_directional_accuracy_improvement_heatmap.png",
    higher_is_better=True,
)


# 4. Comparison table and grouped bars
comparison = build_comparison_table(summary)

for series in SERIES_ORDER:

    plot_grouped_bars(
        comparison=comparison,
        series=series,
        metric_a="standard_rmse_over_sigma",
        metric_b="pe_attention_rmse_over_sigma",
        ylabel="Max-Sigma RMSE/σ",
        title=f"Phase 5 RMSE/σ Comparison: {series_label(series)}",
        filename=f"phase5_{series}_rmse_over_sigma_grouped_bar.png",
    )

    plot_grouped_bars(
        comparison=comparison,
        series=series,
        metric_a="standard_directional_accuracy",
        metric_b="pe_attention_directional_accuracy",
        ylabel="Max-Sigma Directional Accuracy",
        title=f"Phase 5 Directional Accuracy Comparison: {series_label(series)}",
        filename=f"phase5_{series}_directional_accuracy_grouped_bar.png",
    )


# 5. Average improvement by model
by_model = (
    comparison
    .groupby("model")
    .agg(
        avg_rmse_improvement=("rmse_over_sigma_improvement", "mean"),
        avg_direction_improvement=("directional_accuracy_improvement", "mean"),
    )
    .reset_index()
)

by_model["model_label"] = by_model["model"].apply(model_label)

by_model_path = os.path.join(CSV_DIR, "phase5_avg_improvement_by_model.csv")
by_model.to_csv(by_model_path, index=False)

save_grouped_summary_bar(
    df=by_model.sort_values("avg_rmse_improvement", ascending=False),
    category_col="model_label",
    rmse_col="avg_rmse_improvement",
    da_col="avg_direction_improvement",
    title="Phase 5: Average PE-Attention Improvement by Model",
    filename="phase5_avg_improvement_by_model.png",
    xlabel="Model",
)


# 6. Average improvement by time series
by_series = (
    comparison
    .groupby("series")
    .agg(
        avg_rmse_improvement=("rmse_over_sigma_improvement", "mean"),
        avg_direction_improvement=("directional_accuracy_improvement", "mean"),
    )
    .reset_index()
)

by_series["series_label"] = by_series["series"].apply(series_label)

by_series_path = os.path.join(CSV_DIR, "phase5_avg_improvement_by_time_series.csv")
by_series.to_csv(by_series_path, index=False)

save_grouped_summary_bar(
    df=by_series.sort_values("avg_rmse_improvement", ascending=False),
    category_col="series_label",
    rmse_col="avg_rmse_improvement",
    da_col="avg_direction_improvement",
    title="Phase 5: Average PE-Attention Improvement by Time Series",
    filename="phase5_avg_improvement_by_time_series.png",
    xlabel="Time series",
)


# 7. Mean standardized improvement by time series
series_scores = by_series.copy()

rmse_std = series_scores["avg_rmse_improvement"].std(ddof=0)
da_std = series_scores["avg_direction_improvement"].std(ddof=0)

series_scores["rmse_z"] = (
    series_scores["avg_rmse_improvement"]
    - series_scores["avg_rmse_improvement"].mean()
) / (rmse_std + 1e-12)

series_scores["direction_z"] = (
    series_scores["avg_direction_improvement"]
    - series_scores["avg_direction_improvement"].mean()
) / (da_std + 1e-12)

series_scores["mean_standardized_improvement"] = (
    series_scores["rmse_z"] + series_scores["direction_z"]
) / 2

series_score_path = os.path.join(
    CSV_DIR,
    "phase5_mean_standardized_improvement_by_time_series.csv",
)

series_scores.to_csv(series_score_path, index=False)

save_horizontal_bar(
    df=series_scores,
    category_col="series_label",
    value_col="mean_standardized_improvement",
    title="Phase 5: Mean Standardized PE-Attention Improvement by Time Series",
    filename="phase5_mean_standardized_improvement_by_time_series.png",
    xlabel="Mean standardized improvement",
    output_dir=FINAL_DIR,
)


# 8. Ranking by model
model_ranking = by_model.copy()

model_ranking["mean_standardized_improvement"] = (
    (
        model_ranking["avg_rmse_improvement"]
        - model_ranking["avg_rmse_improvement"].mean()
    ) / (model_ranking["avg_rmse_improvement"].std(ddof=0) + 1e-12)
    +
    (
        model_ranking["avg_direction_improvement"]
        - model_ranking["avg_direction_improvement"].mean()
    ) / (model_ranking["avg_direction_improvement"].std(ddof=0) + 1e-12)
) / 2

model_ranking_path = os.path.join(
    CSV_DIR,
    "phase5_average_improvement_ranking_by_model.csv",
)

model_ranking.to_csv(model_ranking_path, index=False)

save_horizontal_bar(
    df=model_ranking,
    category_col="model_label",
    value_col="mean_standardized_improvement",
    title="Phase 5: PE-Attention Robustness Ranking by Model",
    filename="phase5_average_improvement_ranking_by_model.png",
    xlabel="Mean standardized improvement",
    output_dir=RANKING_DIR,
)


# 9. Final interpretive summary
rows = []

for series, group in comparison.groupby("series"):

    best = group.sort_values(
        ["directional_accuracy_improvement", "rmse_over_sigma_improvement"],
        ascending=False,
    ).iloc[0]

    da_gain = best["directional_accuracy_improvement"]
    rmse_gain = best["rmse_over_sigma_improvement"]

    rows.append({
        "stochastic_process": series_label(series),
        "best_model": model_label(best["model"]),
        "da_gain": da_gain,
        "rmse_gain": rmse_gain,
        "conclusion": conclusion_from_gains(da_gain, rmse_gain),
    })

final_summary = pd.DataFrame(rows)

process_order = [
    "Linear Gaussian",
    "Linear Student-t",
    "Nonlinear Gaussian",
    "Nonlinear Student-t",
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
    "phase5_final_interpretive_summary.csv",
)

final_summary.to_csv(final_summary_path, index=False)

save_table_plot(
    df=final_summary,
    title="Phase 5: Best PE-Attention Effect by Stochastic Process",
    filename="phase5_final_interpretive_summary_table.png",
)


print("\nFinished Phase 5 PE-attention plotting.")
print(f"Figures saved in: {FIGURE_DIR}")
print(f"CSV tables saved in: {CSV_DIR}")