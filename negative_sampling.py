import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="processed")
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_user_item_set(train_df):
    return set(zip(train_df["user_idx"].astype(int), train_df["item_idx"].astype(int)))


def negative_sampling(positive_df, num_items, user_item_set, num_negatives=4, seed=42):
    rng = np.random.default_rng(seed)

    users = []
    items = []
    labels = []

    for row in positive_df.itertuples(index=False):
        user = int(row.user_idx)
        pos_item = int(row.item_idx)

        # Positive instance
        users.append(user)
        items.append(pos_item)
        labels.append(1)

        # Negative instances
        sampled = 0
        while sampled < num_negatives:
            neg_item = int(rng.integers(0, num_items))

            if (user, neg_item) not in user_item_set:
                users.append(user)
                items.append(neg_item)
                labels.append(0)
                sampled += 1

    sampled_df = pd.DataFrame({
        "user_idx": users,
        "item_idx": items,
        "label": labels,
    })

    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    train_path = processed_dir / "train.csv"
    positives_path = processed_dir / "train_positives.csv"
    item_mapping_path = processed_dir / "item_mapping.csv"

    train_df = pd.read_csv(train_path)
    positive_df = pd.read_csv(positives_path)
    item_mapping = pd.read_csv(item_mapping_path)

    num_items = item_mapping["item_idx"].nunique()
    user_item_set = build_user_item_set(train_df)

    sampled_df = negative_sampling(
        positive_df=positive_df,
        num_items=num_items,
        user_item_set=user_item_set,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )

    out_path = processed_dir / "train_neg_sampled.csv"
    sampled_df.to_csv(out_path, index=False)

    print("Saved:", out_path.resolve())
    print("Shape:", sampled_df.shape)
    print("Label counts:")
    print(sampled_df["label"].value_counts())


if __name__ == "__main__":
    main()
