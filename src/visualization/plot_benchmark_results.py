import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths

RESULTS_DIR = "results/2026-06-25_full_benchmark"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "summaries",
    "all_models_all_series_summary.csv"
)

FIGURE_DIR ="figures/benchmark_v1"
os.makedirs(FIGURE_DIR, exist_ok=True)


# Constants

COLLAPSE_THRESHOLD = 1.0
CONSECUTIVE_LEVELS = 2
TRAIN_RATIO = 0.7
SERIES_LENGTH = 5000

TRAIN_TEST_SPLIT_INDEX = int(SERIES_LENGTH * TRAIN_RATIO)

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

# Helpers

def load_all_metrics():
    files = glob.glob(os.path.join(METRICS_DIR, "*_metrics.csv"))

    dfs = []

    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_first_crossing(df):
    crossing = df[df["rmse_over_sigma"] >= COLLAPSE_THRESHOLD]

    if crossing.empty:
        return None
    
    return crossing["sigma"].iloc[0]


def get_confirmed_collapse(df):
    df = df.sort_values("sigma").reset_index(drop=True)

    values = df["rmse_over_sigma"].values
    sigmas = df["sigma"].values

    for i in range(len(values) - CONSECUTIVE_LEVELS + 1):
        window = values[i: i + CONSECUTIVE_LEVELS]

        if np.all(window >= COLLAPSE_THRESHOLD):
            return sigmas[i]
        
    return None


def model_label(name):
    return name.replace("_", " ").title()

metrics_df = load_all_metrics()
summary_df = pd.read_csv(SUMMARY_PATH)


# RMSE/SIGMA VS SIGMA

for series in SERIES_ORDER:

    subset = metrics_df[metrics_df["series"] == series]

    plt.figure(figsize=(14,6))

    for model in MODEL_ORDER:
        temp = subset[subset["model"] == model].sort_values("sigma")

        if temp.empty:
            continue

        plt.plot(
            temp["sigma"],
            temp["rmse_over_sigma"],
            marker="o",
            linewidth=1.8,
            label=model_label(model)
        )

        first_cross = get_first_crossing(temp)
        confirmed = get_confirmed_collapse(temp)

        if first_cross is not None:
            y_val = temp[temp["sigma"] == first_cross]["rmse_over_sigma"].iloc[0]
            plt.scatter(first_cross, y_val, s=60)

        if confirmed is not None:
            plt.axvline(
                confirmed,
                linestyle="--",
                linewidth=1.0,
                alpha=0.35
            )

    plt.axhline(
        COLLAPSE_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label="Collapse threshold = 1"
    )

    plt.title(f"Normalized Error Trajectory: {series}", fontsize=16)
    plt.xlabel("Sigma Level")
    plt.ylabel("RmSE / Sigma")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, f"{series}_rmse_over_sigma.png")
    plt.savefig(path, dpi=300)
    plt.close()


# RMSE vs SIGMA

for series in SERIES_ORDER:

    subset = metrics_df[metrics_df["series"] == series]

    plt.figure(figsize=(14,6))

    for model in MODEL_ORDER:
        temp = subset[subset["model"] == model].sort_values("sigma")

        if temp.empty:
            continue

        plt.plot(
            temp["sigma"],
            temp["rmse"],
            marker="o",
            linewidth=1.8,
            label=model_label(model)
        )

    plt.title(f"RMSE Growth Across Noise Levels: {series}", fontsize=16)
    plt.xlabel("Sigma Level")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, f"{series}_rmse.png")
    plt.savefig(path, dpi=300)
    plt.close()


# DIRECTIONAL ACCURACY VS SIGMA

for series in SERIES_ORDER:

    subset = metrics_df[metrics_df["series"] == series]

    plt.figure(figsize=(14,6))

    for model in MODEL_ORDER:
        temp = subset[subset["model"] == model].sort_values("sigma")

        if temp.empty:
            continue

        plt.plot(
            temp["sigma"],
            temp["directional_accuracy"],
            marker="o",
            linewidth=1.8,
            label=model_label(model)
        )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=2,
        label="Random direction = 0.5"
    )

    plt.title(f"Directional Accuracy Across Noise Levels: {series}", fontsize=16)
    plt.xlabel("Sigma Level")
    plt.ylabel("Direcional Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, f"{series}_directional_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.close()


# HEATMAP: LAST RMSE/SIGMA

heatmap_df = summary_df.pivot(
    index="model",
    columns="series",
    values="last_rmse_over_sigma"
).reindex(MODEL_ORDER)

plt.figure(figsize=(10, 9))

plt.imshow(
    heatmap_df.values,
    aspect="auto"
    )

plt.xticks(
    ticks=np.arange(len(heatmap_df.columns)),
    labels=heatmap_df.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(heatmap_df.index)),
    labels=[model_label(m) for m in heatmap_df.index]
)

plt.colorbar(label= "Last RMSE / SIGMA")

for i in range(heatmap_df.shape[0]):
    for j in range(heatmap_df.shape[1]):
        value = heatmap_df.values[i, j]
        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Final Noise-Regime Normalized Error Heatmap", fontsize=16)
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "last_rmse_over_sigma_heatmap.png")
plt.savefig(path, dpi=300)
plt.close()

# Collapse sigma comparison

collapse_df = summary_df.copy()

def collapse_to_numeric(x):
    if isinstance(x, str) and ">" in x:
        return 2.05
    try:
        return float(x)
    except Exception:
        return np.nan
    
collapse_df["collapse_numeric"] = collapse_df["collapse_sigma"].apply(
    collapse_to_numeric
)

for series in SERIES_ORDER:

    subset = collapse_df[collapse_df["series"] == series].copy()
    subset["model"] = pd.Categorical(
        subset["model"],
        categories=MODEL_ORDER,
        ordered=True
    )

    subset = subset.sort_values("model")

    plt.figure(figsize=(14, 7))

    plt.bar(
        [model_label(m) for m in subset["model"]],
        subset["collapse_numeric"]
    )

    plt.axhline(
        2.0,
        linestyle="--",
        linewidth=1.5,
        label="Maximum tested sigma = 2.0"
    )

    plt.title(f"Collapse Sigma Comparison: {series}", fontsize=16)
    plt.xlabel("Model")
    plt.ylabel("Collapse Sigma")
    plt.xticks(rotation=60, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, f"{series}_collapse_sigma_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()


# Average model rank

rank_df = summary_df.copy()

rank_df["rank"] = rank_df.groupby("series")["last_rmse_over_sigma"].rank(
    method="average",
    ascending=True
)

avg_rank = (
    rank_df.groupby("model")["rank"]
    .mean()
    .reindex(MODEL_ORDER)
    .sort_values()
)

plt.figure(figsize=(14,6))

plt.bar(
    [model_label(m) for m in avg_rank.index],
    avg_rank.values
)

plt.title("Average Model Rank Across Stochastic Processes", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Average Rank Lower Is Better")
plt.xticks(rotation=60, ha='right')
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "average_model_rank.png")
plt.savefig(path, dpi=300)
plt.close()


# ACTUAL VS PREDICTED EXMAPLES

EXAMPLE_SERIES = "linear_gaussian"

EXAMPLE_MODELS = [
    "naive",
    "xgboost_increment",
    "lstm_increment",
    "tcn_level",
    "residual_causal_transformer",
    "patchtst_level"
]

plt.figure(figsize=(15, 8))

for model in EXAMPLE_MODELS:

    file_path = os.path.join(
        PREDICTIONS_DIR,
        f"{EXAMPLE_SERIES}_{model}_predictions.csv"
    )

    if not os.path.exists(file_path):
        continue

    pred_df = pd.read_csv(file_path)

    if model == "naive":
        plt.plot(
            pred_df["y_true"].values,
            linewidth=2.5,
            label="Actual"
        )

    plt.plot(
        pred_df["y_pred"].values,
        linewidth=1.5,
        alpha=0.85,
        label=model_label(model)
    )

plt.axvline(
    0,
    linestyle="--",
    linewidth=1.2,
    label="Test region start"
)

plt.title("Actual vs Predicted Values: Linear Gaussian Test Region", fontsize=16)
plt.xlabel("Test Time Index")
plt.ylabel("Series Value")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "prediction_examples_linear_gaussian.png")
plt.savefig(path, dpi=300)
plt.close()


# Absolute error over time

plt.figure(figsize=(15, 8))

for model in EXAMPLE_MODELS:

    file_path = os.path.join(
        PREDICTIONS_DIR,
        f"{EXAMPLE_SERIES}_{model}_predictions.csv"
    )

    if not os.path.exists(file_path):
        continue

    pred_df = pd.read_csv(file_path)

    abs_error = np.abs(pred_df["y_true"] - pred_df["y_pred"])

    plt.plot(
        abs_error.values,
        linewidth=1.5,
        label=model_label(model)
    )

sigma_changes = []
pred_df_ref = pd.read_csv(
    os.path.join(
        PREDICTIONS_DIR,
        f"{EXAMPLE_SERIES}_naive_predictions.csv"
    )
)

sigma_values = pred_df_ref["sigma"].values

for i in range(1, len(sigma_values)):
    if sigma_values[i] != sigma_values[i - 1]:
        sigma_changes.append(i)

for idx in sigma_changes:
    plt.axvline(
        idx,
        linestyle="--",
        linewidth=1.0,
        alpha=0.4
    )

plt.title("Absolute Prediction Error Over Test Time: Linear Gaussian", fontsize=16)
plt.xlabel("Test Time Index")
plt.ylabel("Absolute Error")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "error_vs_time_linear_gaussian.png")
plt.savefig(path, dpi=300)
plt.close()


print(f"All benchmark plots saved to: {FIGURE_DIR}")