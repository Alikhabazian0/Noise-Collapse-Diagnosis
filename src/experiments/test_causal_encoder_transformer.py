import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.causal_encoder_transformer import CausalEncoderTransformerModel
from src.experiments.run_single_model import run_single_model


FILE_PATH = "data/generated/linear_gaussian.csv"

model = CausalEncoderTransformerModel(
    d_model=32,
    nhead=4,
    num_layers=3,
    dim_feedforward=128,
    dropout=0.1,
    epochs=50,
    batch_size=32,
    learning_rate=0.0003
)

evaluation_df, predictions_df, collapse_sigma = run_single_model(
    model=model,
    model_name="causal_encoder_transformer",
    file_path=FILE_PATH,
    save_results=True
)

print("\nEvaluation:")
print(evaluation_df)

print("\nCollapse sigma:")
print(collapse_sigma)