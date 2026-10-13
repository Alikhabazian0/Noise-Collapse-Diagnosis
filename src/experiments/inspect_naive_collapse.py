import pandas as pd

FILES = [
    "results/metrics/linear_gaussian_naive_metrics.csv",
    "results/metrics/linear_student_t_naive_metrics.csv",
    "results/metrics/nonlinear_gaussian_naive_metrics.csv",
    "results/metrics/nonlinear_student_t_naive_metrics.csv"
]

for file in FILES:

    print("\n" + "=" * 80)

    df = pd.read_csv(file)

    print(df[[
        "sigma",
        "rmse",
        "rmse_over_sigma",
        "directional_accuracy"
    ]])

    print("=" * 80)