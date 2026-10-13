import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.simple_transformer import SimpleCausalTransformerModel
from src.experiments.run_single_model import run_single_model


FILE_PATH = "data/generated/linear_gaussian.csv"

model = SimpleCausalTransformerModel(
    d_model=16,
    nhead=2,
    num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    epochs=100,
    batch_size=32,
    learning_rate=0.0001
)

evaluation_df, predictions_df, collapse_sigma = run_single_model(
    model=model,
    model_name="simple_causal_transformer",
    file_path=FILE_PATH,
    save_results=True
)

print("\nEvaluation:")
print(evaluation_df)

print("\nCollapse sigma:")
print(collapse_sigma)