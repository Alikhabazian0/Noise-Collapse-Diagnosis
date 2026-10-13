import numpy as np
import pandas as pd

from src.evaluation.metrics import rmse, directional_accuracy

def evaluate_by_sigma(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_values: np.ndarray
) -> pd.DataFrame:
    """
    Compute RMSE, RMSE/sigma, and directional accuracy for each sigma level.
    """

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "sigma": sigma_values
    })

    rows = []

    for sigma, group in df.groupby("sigma"):

        sigma_rmse = rmse(
            group["y_true"].values,
            group["y_pred"].values
        )

        if sigma == 0:
            rmse_over_sigma = np.nan
        else:
            rmse_over_sigma = sigma_rmse / sigma
        
        sigma_da = directional_accuracy(
            group["y_true"].values,
            group["y_pred"].values
        )

        rows.append({
            "sigma": sigma,
            "n_samples": len(group),
            "rmse": sigma_rmse,
            "rmse_over_sigma": rmse_over_sigma,
            "directional_accuracy": sigma_da
            })
        
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)

def find_collapse_point(
        evaluation_df: pd.DataFrame,
        threshold: float = 1.0,
        consecutive_levels: int = 2 ##
):
    """
    Collapse point = first sigma where RMSE/sigma >= threshold
    (and remains above threshold for all later sigma levels. - unstable, ver 0)
    for a required number of consecutive sigma levels. (stable, ver 1 - current)
    
    Default:
    - threshold = 1.0
    - consecutive_levels = 2
    """

    valid_df = evaluation_df.dropna(subset=["rmse_over_sigma"]).copy()
    valid_df = valid_df.sort_values("sigma").reset_index(drop=True) ## v1 added

# v0 logic
#    for i in range(len(valid_df)):
#
#        current_and_after = valid_df.iloc[i:]
#
#        if (current_and_after["rmse_over_sigma"] >= threshold).all():
#            return valid_df.iloc[i]["sigma"]
#        
#    return np.inf

    ratios = valid_df["rmse_over_sigma"].values
    sigmas = valid_df["sigma"].values

    for i in range(len(ratios) - consecutive_levels + 1):

        window = ratios[i:i + consecutive_levels]

        if (window >= threshold).all():
            return sigmas[i]

    return None