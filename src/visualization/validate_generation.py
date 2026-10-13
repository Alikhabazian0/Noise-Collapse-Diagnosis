import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "data/generated"
OUTPUT_DIR = "figures/data_generation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "linear_gaussian.csv",
    "linear_student_t.csv",
    "nonlinear_gaussian.csv",
    "nonlinear_student_t.csv"
]


# PLOT 1 - SERIES OVERVIEW
fig, axes = plt.subplots(2,2,figsize=(14,6))

for ax, file in zip(axes.flatten(), FILES):

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    ax.plot(df["x"])
    ax.set_title(file.replace(".csv",""))

plt.tight_layout()

plt.title("Generated Time Series")

plt.savefig(
    f"{OUTPUT_DIR}/series_overview.png",
    dpi=300
)

plt.close()

# PLOT 2 - SIGMA PATHS
fig, axes = plt.subplots(2,2,figsize=(14,6))

for ax, file in zip(axes.flatten(), FILES):

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    ax.plot(df["sigma"])
    ax.set_title(file.replace(".csv",""))

plt.tight_layout()

plt.title("Segment Aware Sigma Paths")

plt.savefig(
    f"{OUTPUT_DIR}/sigma_paths.png",
    dpi=300
)

plt.close()

# PLOT 3 - Noise Histograms
plt.figure(figsize=(14,6))

for file in FILES:

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    plt.hist(
        df["noise"],
        bins=80,
        density=True,
        alpha=0.4,
        label=file.replace(".csv", "")
    )

plt.legend()

plt.xlim(-8,8)

plt.title("Noise Distribution Comparison")

plt.savefig(
    f"{OUTPUT_DIR}/noise_histograms.png",
    dpi=300
)

plt.close()

# PLOT 4 - SEGMENT VOLATILITY
results = []

for file in FILES:

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    returns = np.diff(df["x"])

    segments = np.array_split(returns, 10)

    vol = [np.std(seg) for seg in segments]

    results.append(vol)

plt.figure(figsize=(14,6))

for file, vol in zip(FILES, results):

    plt.plot(
        range(1,11),
        vol,
        marker="o",
        label=file.replace(".csv", "")
    )

plt.xlabel("segments")

plt.ylabel("Volatility")

plt.legend()

plt.title("segment volatility")

plt.savefig(
    f"{OUTPUT_DIR}/segment_volatility.png",
    dpi=300
)

plt.close()

# PLOT 5 - NOISE-TO-TREND RATIO
plt.figure(figsize=(14,6))

for file in FILES:

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    segment_size = len(df) // 10

    ratios = []

    for i in range(10):

        start = i * segment_size

        if i == 9:
            end = len(df)
        else:
            end = (i + 1) * segment_size
        seg = df.iloc[start:end]

        sigma_mean = seg["sigma"].mean()
        trend_mean = np.abs(seg["trend"]).mean()
        
        ratios.append(
            sigma_mean / (trend_mean + 1e-12)
        )

    plt.plot(
        range(1,11),
        ratios,
        marker="o",
        label=file.replace(".csv", "")
    )

plt.xlabel("Segment")

plt.ylabel("Noise / Trend Ratio")

plt.legend()

plt.title("noise to trend ratio")

plt.savefig(
    f"{OUTPUT_DIR}/noise_to_trend_ratio.png",
    dpi=300
)

plt.close()

# Save validation summary tables
summary_rows = []

for file in FILES:

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    returns = df["x"].diff().dropna().reset_index(drop=True)

    segment_size = len(df) // 10

    for i in range(10):

        start = i * segment_size

        if i == 9:
            end = len(df)
        else:
            end = (i + 1) * segment_size

        seg = df.iloc[start:end]

        seg_returns = returns.iloc[start:end - 1]

        summary_rows.append({
            "series": file.replace(".csv", ""),
            "segment": i + 1,
            "mean_sigma": seg["sigma"].mean(),
            "std_returns": seg_returns.std(),
            "mean_abs_trend": seg["trend"].abs().mean(),
            "noise_to_trend_ratio": seg["sigma"].mean() / (seg["trend"].abs().mean() + 1e-12),
            "mean_x": seg["x"].mean(),
            "std_x": seg["x"].std()
        })

summary_df = pd.DataFrame(summary_rows)

summary_path = "results/generation_validation_summary.csv"

os.makedirs("results", exist_ok=True)

summary_df.to_csv(summary_path, index=False)


print(f"Saved validation summary: {summary_path}")
print("Validation plots saved.")