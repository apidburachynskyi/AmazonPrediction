import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_positives", type=int, default=0, help="0 = all positives")
    return parser.parse_args()


def build_seen_items(train, val, test):
    all_interactions = pd.concat([train, val, test], ignore_index=True)
    seen = {}

    for row in all_interactions[["user_idx", "item_idx"]].itertuples(index=False):
        user = int(row.user_idx)
        item = int(row.item_idx)
        seen.setdefault(user, set()).add(item)

    return seen


def sample_negatives_for_user(user, gt_item, num_items, seen_items, num_negatives, rng):
    negatives = []
    seen = seen_items.get(user, set())

    while len(negatives) < num_negatives:
        item = int(rng.integers(0, num_items))

        if item == gt_item:
            continue

        if item in seen:
            continue

        negatives.append(item)

    return negatives


def ndcg_at_rank(rank):
    return 1.0 / math.log2(rank + 2)


def evaluate_random_split(
    split_df,
    num_items,
    seen_items,
    k,
    num_negatives,
    seed,
    max_eval_positives,
):
    rng = np.random.default_rng(seed)

    positives = split_df[split_df["label"] == 1][["user_idx", "item_idx"]].copy()

    if max_eval_positives and len(positives) > max_eval_positives:
        positives = positives.sample(n=max_eval_positives, random_state=seed).reset_index(drop=True)

    hits = []
    ndcgs = []

    for row in positives.itertuples(index=False):
        user = int(row.user_idx)
        gt_item = int(row.item_idx)

        negatives = sample_negatives_for_user(
            user=user,
            gt_item=gt_item,
            num_items=num_items,
            seen_items=seen_items,
            num_negatives=num_negatives,
            rng=rng,
        )

        candidate_items = negatives + [gt_item]
        scores = rng.random(len(candidate_items))
        item_score = dict(zip(candidate_items, scores))

        ranklist = sorted(candidate_items, key=lambda item: item_score[item], reverse=True)[:k]

        if gt_item in ranklist:
            rank = ranklist.index(gt_item)
            hits.append(1.0)
            ndcgs.append(ndcg_at_rank(rank))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

    return {
        f"HR@{k}": float(np.mean(hits)) if hits else None,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        "num_eval_positives": int(len(positives)),
        "num_negatives": int(num_negatives),
    }


def main():
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    item_mapping = pd.read_csv(processed_dir / "item_mapping.csv")

    num_items = item_mapping["item_idx"].nunique()
    seen_items = build_seen_items(train, val, test)

    val_metrics = evaluate_random_split(
        split_df=val,
        num_items=num_items,
        seen_items=seen_items,
        k=args.k,
        num_negatives=args.num_negatives,
        seed=args.seed,
        max_eval_positives=args.max_eval_positives,
    )

    test_metrics = evaluate_random_split(
        split_df=test,
        num_items=num_items,
        seen_items=seen_items,
        k=args.k,
        num_negatives=args.num_negatives,
        seed=args.seed + 1,
        max_eval_positives=args.max_eval_positives,
    )

    row = {
        "model": "Random",
        f"Val HR@{args.k}": val_metrics[f"HR@{args.k}"],
        f"Val NDCG@{args.k}": val_metrics[f"NDCG@{args.k}"],
        f"Test HR@{args.k}": test_metrics[f"HR@{args.k}"],
        f"Test NDCG@{args.k}": test_metrics[f"NDCG@{args.k}"],
        "val_num_eval_positives": val_metrics["num_eval_positives"],
        "test_num_eval_positives": test_metrics["num_eval_positives"],
        "num_negatives": args.num_negatives,
    }

    print(json.dumps(row, indent=4))

    out_path = output_dir / f"random_baseline_k{args.k}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    # Also save with the old expected name, so existing collect scripts can still read it.
    compatibility_path = output_dir / f"baseline_ranking_k{args.k}.csv"
    pd.DataFrame([row]).to_csv(compatibility_path, index=False)

    print("\nSaved random baseline to:", out_path)
    print("Saved compatibility copy to:", compatibility_path)


if __name__ == "__main__":
    main()
