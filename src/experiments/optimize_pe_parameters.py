import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.evaluation.entropy import permutation_entropy


SERIES_FILES = [
    "data/generated/linear_gaussian.csv",
    "data/generated/linear_student_t.csv",
    "data/generated/nonlinear_gaussian.csv",
    "data/generated/nonlinear_student_t.csv",
]

OUTPUT_DIR = "results/pe_transformer_experiment/pe_optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_GRID = [8, 12, 16]
ORDER_GRID = [3, 4, 5]
DELAY_GRID = [1, 2]

VALUE_COLUMN = "x"
SIGMA_COLUMN = "sigma"

LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
EPS = 1e-8


def compute_rolling_permutation_entropy(values, window, order, delay):
    pe_values = np.full(len(values), np.nan)

    for i in range(window - 1, len(values)):
        segment = values[i - window + 1:i + 1]

        pe_values[i] = permutation_entropy(
            segment,
            order=order,
            delay=delay,
            normalize=True
        )

    return pe_values


def score_pe_separation(df, pe_column):
    low_threshold = df[SIGMA_COLUMN].quantile(LOW_QUANTILE)
    high_threshold = df[SIGMA_COLUMN].quantile(HIGH_QUANTILE)

    low_group = df[df[SIGMA_COLUMN] <= low_threshold][pe_column].dropna()
    high_group = df[df[SIGMA_COLUMN] >= high_threshold][pe_column].dropna()

    if len(low_group) < 20 or len(high_group) < 20:
        return None

    low_mean = low_group.mean()
    high_mean = high_group.mean()
    low_std = low_group.std()
    high_std = high_group.std()

    score = abs(high_mean - low_mean) / (low_std + high_std + EPS)

    return {
        "score": float(score),
        "low_mean_pe": float(low_mean),
        "high_mean_pe": float(high_mean),
        "low_std_pe": float(low_std),
        "high_std_pe": float(high_std),
        "low_n": int(len(low_group)),
        "high_n": int(len(high_group)),
    }


all_results = []

for file_path in SERIES_FILES:

    df = pd.read_csv(file_path)
    series_name = os.path.basename(file_path).replace(".csv", "")
    values = df[VALUE_COLUMN].values.astype(float)

    print("\n" + "=" * 80)
    print(f"Optimizing PE parameters for: {series_name}")
    print("=" * 80)

    for window, order, delay in itertools.product(WINDOW_GRID, ORDER_GRID, DELAY_GRID):

        if window <= order * delay:
            print(f"Skipping invalid: window={window}, order={order}, delay={delay}")
            continue

        pe_values = compute_rolling_permutation_entropy(
            values=values,
            window=window,
            order=order,
            delay=delay
        )

        temp_df = df.copy()
        temp_df["rolling_pe"] = pe_values

        score_info = score_pe_separation(
            df=temp_df,
            pe_column="rolling_pe"
        )

        if score_info is None:
            print(f"Skipping insufficient data: window={window}, order={order}, delay={delay}")
            continue

        row = {
            "series": series_name,
            "window": window,
            "order": order,
            "delay": delay,
            **score_info
        }

        all_results.append(row)

        print(
            f"window={window}, order={order}, delay={delay} | "
            f"score={score_info['score']:.6f}"
        )


all_results_df = pd.DataFrame(all_results)

if all_results_df.empty:
    raise RuntimeError("No valid PE parameter combinations found for any series.")

best_params_df = (
    all_results_df
    .loc[all_results_df.groupby("series")["score"].idxmax()]
    .rename(columns={
        "window": "best_window",
        "order": "best_order",
        "delay": "best_delay",
        "score": "best_score"
    })
    .reset_index(drop=True)
)

all_results_path = os.path.join(
    OUTPUT_DIR,
    "pe_parameter_optimization_all_results.csv"
)

best_params_path = os.path.join(
    OUTPUT_DIR,
    "best_pe_params_by_series.csv"
)

all_results_df.to_csv(all_results_path, index=False)
best_params_df.to_csv(best_params_path, index=False)

print("\n" + "=" * 80)
print("PE PARAMETER OPTIMIZATION FINISHED")
print("=" * 80)
print(f"Saved all results: {all_results_path}")
print(f"Saved best params: {best_params_path}")
print("\nBest parameters by series:")
print(best_params_df)