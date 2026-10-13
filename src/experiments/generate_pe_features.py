import os
import sys
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

BEST_PARAMS_PATH = (
    "results/pe_transformer_experiment/"
    "pe_optimization/best_pe_params_by_series.csv"
)

OUTPUT_DIR = "results/pe_transformer_experiment/pe_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALUE_COLUMN = "x"
SIGMA_COLUMN = "sigma"


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


best_params_df = pd.read_csv(BEST_PARAMS_PATH)

all_feature_rows = []

for file_path in SERIES_FILES:

    df = pd.read_csv(file_path)
    series_name = os.path.basename(file_path).replace(".csv", "")

    params = best_params_df[
        best_params_df["series"] == series_name
    ].iloc[0]

    window = int(params["best_window"])
    order = int(params["best_order"])
    delay = int(params["best_delay"])

    print("\n" + "=" * 80)
    print(f"Generating PE feature for: {series_name}")
    print(f"window={window}, order={order}, delay={delay}")
    print("=" * 80)

    values = df[VALUE_COLUMN].values.astype(float)

    rolling_pe = compute_rolling_permutation_entropy(
        values=values,
        window=window,
        order=order,
        delay=delay
    )

    feature_df = df.copy()
    feature_df["time_index"] = np.arange(len(feature_df))
    feature_df["pe_feature"] = rolling_pe
    feature_df["pe_window"] = window
    feature_df["pe_order"] = order
    feature_df["pe_delay"] = delay
    feature_df["series"] = series_name

    # Fill initial NaN values without future leakage:
    # first valid PE is copied backward only for initialization.
    first_valid = feature_df["pe_feature"].first_valid_index()

    if first_valid is not None:
        feature_df["pe_feature"] = feature_df["pe_feature"].fillna(0.0)
    else:
        raise RuntimeError(f"No valid PE values generated for {series_name}")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{series_name}_with_pe_features.csv"
    )

    feature_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    all_feature_rows.append(feature_df)


combined_df = pd.concat(all_feature_rows, ignore_index=True)

combined_path = os.path.join(
    OUTPUT_DIR,
    "all_series_with_pe_features.csv"
)

combined_df.to_csv(combined_path, index=False)

print("\n" + "=" * 80)
print("PE FEATURE GENERATION FINISHED")
print("=" * 80)
print(f"Saved combined file: {combined_path}")
print("\nPreview:")
print(combined_df.head())