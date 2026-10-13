import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.naive import NaivePersistence
from src.experiments.run_single_model import run_single_model

FILES = [
    "data/generated/linear_gaussian.csv",
    "data/generated/linear_student_t.csv",
    "data/generated/nonlinear_gaussian.csv",
    "data/generated/nonlinear_student_t.csv"    
]

all_results = []

for file_path in FILES:

    print(f"\nRunning: {file_path}")

    model = NaivePersistence()

    evaluation_df, prediction_df, collapse_sigma = run_single_model(
        model=model,
        model_name="naive",
        file_path=file_path,
        save_results=True
    )

    summary_row = {
        "series": evaluation_df["series"].iloc[0],
        "model": "naive",
    #    "collapse_sigma": collapse_sigma, - v0
        "collapse_sigma": collapse_sigma if collapse_sigma is not None else "> 2.0",        # v1
        "collapse_status": "confirmed" if collapse_sigma is not None else "not confirmed",  # v1
        "last_rmse": evaluation_df["rmse"].iloc[-1],
        "last_rmse_over_sigma": evaluation_df["rmse_over_sigma"].iloc[-1],
        "last_directional_accuracy": evaluation_df["directional_accuracy"].iloc[-1]
    }

    all_results.append(summary_row)

summary_df = pd.DataFrame(all_results)

os.makedirs("results/summaries", exist_ok=True)

summary_path = "results/summaries/naive_summary.csv"

summary_df.to_csv(summary_path, index=False)

print("\n================")
print("NAIVE SUMMARY")
print("================")
print(summary_df)

print(f"\nSaved: {summary_path}")