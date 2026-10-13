from src.models.simple_transformer import SimpleCausalTransformerModel

class CausalEncoderTransformerModel(SimpleCausalTransformerModel):
    """
    Stronger causal encoder transformer.

    Difference from simple transformer:
    - more encoder layers
    - larger feedforward block
    - same causal mask
    - same level prediction task
    """

    def __init__(
            self,
            input_size=1,
            d_model=32,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1,
            epochs=50,
            batch_size=32,
            learning_rate=0.0003,
            device=None
    ):
        super().__init__(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )