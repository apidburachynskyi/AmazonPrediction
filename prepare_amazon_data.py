"""
prepare_amazon_data.py

Prepare Amazon Reviews 2023 data for Neural Collaborative Filtering.

Usage:
    python prepare_amazon_data.py --input All_Beauty.jsonl --nrows 100000 --out_dir processed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to Amazon Reviews jsonl file.")
    parser.add_argument("--out_dir", type=str, default="processed", help="Output directory.")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to read. Use None for full file.")
    parser.add_argument("--min_user_interactions", type=int, default=2)
    parser.add_argument("--min_item_interactions", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def load_raw_data(path, nrows=None):
    print(f"Loading data from: {path}")
    df = pd.read_json(path, lines=True, nrows=nrows)

    required_cols = ["user_id", "parent_asin", "rating", "timestamp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {df.columns.tolist()}")

    df = df[required_cols].copy()
    df = df.rename(columns={"parent_asin": "item_id"})
    return df


def create_binary_labels(df):
    df["rating"] = df["rating"].astype(float)

    # Remove neutral ratings
    df = df[df["rating"] != 3.0].copy()

    # 4/5 positive, 1/2 negative
    df["label"] = (df["rating"] >= 4.0).astype(int)

    return df


def filter_interactions(df, min_user_interactions=3, min_item_interactions=3):
    """
    Filter users/items with too few interactions.
    Repeated because after filtering users, item counts can change and vice versa.
    """
    prev_shape = None

    while prev_shape != df.shape:
        prev_shape = df.shape

        user_counts = df["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(keep_users)].copy()

        item_counts = df["item_id"].value_counts()
        keep_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(keep_items)].copy()

    return df


def encode_ids(df):
    user_idx, user_uniques = pd.factorize(df["user_id"])
    item_idx, item_uniques = pd.factorize(df["item_id"])

    df["user_idx"] = user_idx.astype(np.int64)
    df["item_idx"] = item_idx.astype(np.int64)

    user_mapping = pd.DataFrame({
        "user_idx": np.arange(len(user_uniques)),
        "user_id": user_uniques,
    })

    item_mapping = pd.DataFrame({
        "item_idx": np.arange(len(item_uniques)),
        "item_id": item_uniques,
    })

    return df, user_mapping, item_mapping


def split_data(df, random_state=42):
    """
    Simple stratified random split.
    Later, for ranking evaluation, we can replace it with leave-one-out.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    return train_df, val_df, test_df


def save_outputs(df, train_df, val_df, test_df, user_mapping, item_mapping, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cols = ["user_idx", "item_idx", "label"]

    train_df[cols].to_csv(out_path / "train.csv", index=False)
    val_df[cols].to_csv(out_path / "val.csv", index=False)
    test_df[cols].to_csv(out_path / "test.csv", index=False)

    # Useful for negative sampling from positive train interactions
    train_df[train_df["label"] == 1][["user_idx", "item_idx"]].to_csv(
        out_path / "train_positives.csv",
        index=False,
    )

    user_mapping.to_csv(out_path / "user_mapping.csv", index=False)
    item_mapping.to_csv(out_path / "item_mapping.csv", index=False)

    stats = []
    stats.append(f"Full processed shape: {df.shape}")
    stats.append(f"Train shape: {train_df.shape}")
    stats.append(f"Validation shape: {val_df.shape}")
    stats.append(f"Test shape: {test_df.shape}")
    stats.append(f"Number of users: {df['user_idx'].nunique()}")
    stats.append(f"Number of items: {df['item_idx'].nunique()}")
    stats.append("Label distribution:")
    stats.append(str(df["label"].value_counts()))
    stats.append("Label distribution normalized:")
    stats.append(str(df["label"].value_counts(normalize=True)))

    with open(out_path / "dataset_stats.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(stats))

    print("\n".join(stats))
    print(f"\nSaved files to: {out_path.resolve()}")


def main():
    args = parse_args()

    df = load_raw_data(args.input, args.nrows)
    print("Raw shape:", df.shape)

    df = create_binary_labels(df)
    print("After binary labels:", df.shape)

    df = filter_interactions(
        df,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    print("After filtering:", df.shape)

    df, user_mapping, item_mapping = encode_ids(df)
    train_df, val_df, test_df = split_data(df, random_state=args.random_state)

    save_outputs(df, train_df, val_df, test_df, user_mapping, item_mapping, args.out_dir)


if __name__ == "__main__":
    main()
