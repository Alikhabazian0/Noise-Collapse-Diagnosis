import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.residual_transformer import ResidualCausalTransformerModel
from src.experiments.run_single_model import run_single_model


FILE_PATH = "data/generated/linear_gaussian.csv"

model = ResidualCausalTransformerModel(
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    dropout=0.0,
    epochs=50,
    batch_size=32,
    learning_rate=0.0003
)

evaluation_df, predictions_df, collapse_sigma = run_single_model(
    model=model,
    model_name="residual_causal_transformer",
    file_path=FILE_PATH,
    save_results=True
)

print("\nEvaluation:")
print(evaluation_df)

print("\nCollapse sigma:")
print(collapse_sigma)