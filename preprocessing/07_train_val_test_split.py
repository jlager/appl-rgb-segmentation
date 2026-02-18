import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


METADATA_DIR = os.path.join(os.getcwd(), "data", "raw", "metadata")
METADATA_NAME = "metadata.csv"
SAVE_DIR = os.path.join(os.getcwd(), "data", "metadata")
MAX_RESHUFFLE_ITERS = 1000
SAMPLES_PER_SPLIT = 250
RANDOM_STATE = 42


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and target sets."""
    splitter = GroupShuffleSplit(
        n_splits=MAX_RESHUFFLE_ITERS,
        test_size=SAMPLES_PER_SPLIT / len(df),
        random_state=RANDOM_STATE,
    )

    for train_idx, target_idx in splitter.split(df, groups=df["Group"]):
        if len(df.iloc[target_idx]) == SAMPLES_PER_SPLIT:
            break

    if len(df.iloc[target_idx]) != SAMPLES_PER_SPLIT:
        raise ValueError("Failed to find exact split")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    target_df = df.iloc[target_idx].reset_index(drop=True)
    return train_df, target_df


def main() -> None:
    df = pd.read_csv(os.path.join(METADATA_DIR, METADATA_NAME))
    df["Group"] = df["Species"].astype(str) + "_" + df["Plant ID"].astype(str)

    train_df, test_df = split(df)
    train_df, val_df = split(train_df)

    assert set(train_df["Group"]).isdisjoint(set(val_df["Group"]))
    assert set(train_df["Group"]).isdisjoint(set(test_df["Group"]))
    assert set(val_df["Group"]).isdisjoint(set(test_df["Group"]))

    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SAVE_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SAVE_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SAVE_DIR, "test.csv"), index=False)


if __name__ == "__main__":
    main()
