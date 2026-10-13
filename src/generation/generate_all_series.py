import os
import sys
import pandas as pd
import numpy as np

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.config import (
    RANDOM_SEED,
    SERIES_LENGTH,
    DT,
    SIGMA_LEVELS,
    TREND_TYPES,
    NOISE_TYPES,
    LINEAR_DRIFT,
    NONLINEAR_DRIFT,
    NONLINEAR_AMPLITUDE,
    NONLINEAR_PERIOD,
    STUDENT_T_DF
)

from src.generation.trends import get_trend
from src.generation.noise import get_noise
from src.generation.euler_maruyama import (
    create_stepwise_sigma_path,
    simulate_euler_maruyama
)

def generate_single_series(
        trend_type: str,
        noise_type: str,
        seed: int
) -> pd.DataFrame:
    """
    Generate one synthetic Euler-Maruyama time series.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(SERIES_LENGTH)

    trend = get_trend(
        trend_type=trend_type,
        t=t,
        linear_drift=LINEAR_DRIFT,
        nonlinear_drift=NONLINEAR_DRIFT,
        nonlinear_amplitude=NONLINEAR_AMPLITUDE,
        nonlinear_period=NONLINEAR_PERIOD
    )

    noise = get_noise(
        noise_type=noise_type,
        size=SERIES_LENGTH,
        rng=rng,
        student_t_df=STUDENT_T_DF
    )

    sigma_path = create_stepwise_sigma_path(
        series_length=SERIES_LENGTH,
        sigma_levels=SIGMA_LEVELS
    )

    x = simulate_euler_maruyama(
        trend=trend,
        noise=noise,
        sigma_path=sigma_path,
        dt=DT,
        x0=0.0
    )

    df = pd.DataFrame({
        "t": t,
        "x": x,
        "trend": trend,
        "noise": noise,
        "sigma": sigma_path,
        "trend_type": trend_type,
        "noise_type": noise_type
    })

    return df

def generate_all_series():
    """
    Generate all 4 combinations:
    linear/nonlinear trend * gaussian/student-t (non-gaussian) noise.
    """
    output_dir = "data/generated"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    counter = 0

    for trend_type in TREND_TYPES:
        for noise_type in NOISE_TYPES:
            seed = RANDOM_SEED + counter

            df = generate_single_series(
                trend_type=trend_type,
                noise_type=noise_type,
                seed=seed
            )

            filename = f"{trend_type}_{noise_type}.csv"
            path = os.path.join(output_dir, filename)

            df.to_csv(path, index=False)

            print(f"Saved: {path}")

            all_data.append(df)
            counter += 1

    combined = pd.concat(all_data, ignore_index=True)
    combined_path = os.path.join(output_dir, "all_series.csv")
    combined.to_csv(combined_path, index=False)

    print(f"Saved combined file: {combined_path}")

if __name__ == "__main__":
    generate_all_series()