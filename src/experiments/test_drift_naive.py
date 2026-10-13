import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.naive import DriftNaive
from src.experiments.run_single_model import run_single_model

FILE_PATH = "data/generated/linear_gaussian.csv"

model = DriftNaive()

evaluation_df, predictions_df, collapse_sigma = run_single_model(
    model = model,
    model_name="drift_naive",
    file_path=FILE_PATH,
    save_results=True
)

print("\nEvaluation:")
print(evaluation_df)

print("\nCollapse sigma:")
print(collapse_sigma)