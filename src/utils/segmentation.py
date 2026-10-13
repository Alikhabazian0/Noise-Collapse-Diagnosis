import numpy as np
import pandas as pd


def get_train_test_split_index(full_predictions_df: pd.DataFrame) -> int:
    """
    Return the first time_index belonging to the test split.
    """
    test_df = full_predictions_df[full_predictions_df["split"] == "test"]

    if test_df.empty:
        raise ValueError("No test split found in full_predictions_df.")
    
    return int(test_df["time_index"].min())


def get_collapse_index(
    full_predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    collapse_threshold: float = 1.0
) -> int:
    """
    Find the first test-set time_index where the model collapses.

    Collapse is identified from metrics_df by the first sigma where
    rmse_over_sigma >= collapse_threshold.

    Then the collapse sigma is mapped only inside the test split,
    because collapse is evaluated on test performance.
    """

    metrics_df = metrics_df.sort_values("sigma").reset_index(drop=True)

    collapsed = metrics_df[
        metrics_df["rmse_over_sigma"] >= collapse_threshold
    ]

    test_df = full_predictions_df[
        full_predictions_df["split"] == "test"
    ].copy()

    if test_df.empty:
        raise ValueError("No test split found in full_predictions_df.")

    if collapsed.empty:
        # If no collapse occurs, use the final available test time index.
        return int(test_df["time_index"].max())

    collapse_sigma = collapsed["sigma"].iloc[0]

    matching = test_df[
        test_df["sigma"] == collapse_sigma
    ]

    if matching.empty:
        raise ValueError(
            f"No matching test sigma={collapse_sigma} found in full predictions."
        )

    return int(matching["time_index"].min())


def build_precollapse_segments(
        full_predictions_df : pd.DataFrame,
        collapse_index : int,
        n_segments: int = 10
) -> pd.DataFrame:
    """
    build non-overlapping segments from the beginning of the available 
    supervised trajectory until the collapse index.

    each segment covers approximately 1/n of the interval:
    [min_time_index, collapse_index]
    
    Returns a DataFrame with:
    - segment
    - progress_start
    - progress_end
    - start_index
    - end_index
    - split_status
    - n_samples
    """

    min_index = int(full_predictions_df["time_index"].min())

    if collapse_index <= min_index:
        raise ValueError("collapse_index must be larger than min time_index.")
    
    edges = np.linspace(
        min_index,
        collapse_index,
        n_segments + 1
    )

    rows = []

    for i in range(n_segments):

        start = int(np.floor(edges[i]))
        end = int(np.floor(edges[i+1]))

        if i == n_segments - 1:
            end = collapse_index
            
        segment_df = full_predictions_df[
            (full_predictions_df["time_index"] >= start)
            & (full_predictions_df["time_index"] <= end)
        ]

        if segment_df.empty:
            split_status = "empty"
            n_samples = 0
        else:
            unique_splits = segment_df["split"].unique()

            if len(unique_splits) == 1:
                split_status = unique_splits[0]
            else:
                split_status = "mixed"

            n_samples = len(segment_df)

        rows.append({
            "segment": i + 1,
            "progress_start": i / n_segments,
            "progress_end": (i + 1) / n_segments,
            "progress_mid": (i + 0.5) / n_segments,
            "start_index": start,
            "end_index": end,
            "collapse_index": collapse_index,
            "split_status": split_status,
            "n_samples": n_samples
        })

    return pd.DataFrame(rows)


def attach_segment_labels(
    full_predictions_df: pd.DataFrame,
    segments_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach a segment label to each row of full_predictions_df.
    Rows outside all segments receive NaN.
    """

    df = full_predictions_df.copy()
    df["segment"] = np.nan

    for _, row in segments_df.iterrows():
        mask = (
            (df["time_index"] >= row["start_index"])
            & (df["time_index"] <= row["end_index"])
        )

        df.loc[mask, "segment"] = int(row["segment"])

    return df