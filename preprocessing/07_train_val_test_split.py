import os
from typing import Tuple

import numpy as np
import pandas as pd


METADATA_DIR = os.path.join(os.getcwd(), "data", "raw", "metadata")
METADATA_NAME = "metadata.csv"
SAVE_DIR = os.path.join(os.getcwd(), "data", "metadata")
MAX_RESHUFFLE_ITERS = 1000
SAMPLES_PER_SPLIT = 250
RANDOM_STATE = 42


def artifact_group(file_name: str) -> str:
    """Return the experiment/plant group encoded in an artifact key."""
    if not file_name.endswith(".png"):
        raise ValueError(f"Artifact key must end in .png: {file_name}")

    parts = file_name[:-4].split("__")
    if len(parts) != 5 or any(not part for part in parts):
        raise ValueError(f"Invalid artifact key: {file_name}")
    return f"{parts[0]}__{parts[2]}"


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and target sets."""
    _, group_indices = np.unique(df["Split Group"].to_numpy(), return_inverse=True)
    n_groups = group_indices.max() + 1
    n_target_groups = int(np.ceil(SAMPLES_PER_SPLIT / len(df) * n_groups))
    rng = np.random.RandomState(RANDOM_STATE)

    for _ in range(MAX_RESHUFFLE_ITERS):
        target_groups = rng.permutation(n_groups)[:n_target_groups]
        target_mask = np.isin(group_indices, target_groups)
        target_idx = np.flatnonzero(target_mask)
        if len(target_idx) == SAMPLES_PER_SPLIT:
            train_idx = np.flatnonzero(~target_mask)
            break
    else:
        raise ValueError("Failed to find exact split")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    target_df = df.iloc[target_idx].reset_index(drop=True)
    return train_df, target_df


def main() -> None:
    df = pd.read_csv(os.path.join(METADATA_DIR, METADATA_NAME))

    train_df, test_df = split(df)
    train_df, val_df = split(train_df)

    for split_df in (train_df, val_df, test_df):
        split_df["Group"] = split_df["File Name"].map(artifact_group)
        split_df.drop(columns=["Split Group"], inplace=True)

    assert set(train_df["Group"]).isdisjoint(set(val_df["Group"]))
    assert set(train_df["Group"]).isdisjoint(set(test_df["Group"]))
    assert set(val_df["Group"]).isdisjoint(set(test_df["Group"]))

    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SAVE_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SAVE_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SAVE_DIR, "test.csv"), index=False)


if __name__ == "__main__":
    main()
