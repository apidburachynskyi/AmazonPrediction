import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["gmf", "mlp", "neumf", "all"])
    parser.add_argument("--processed_dir", type=str, default="Data/Processed")
    parser.add_argument("--runs_dir", type=str, default="Runs/BCE")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_users", type=int, default=None,
                        help="Optional limit for faster debugging.")
    return parser.parse_args()


def hit_ratio(ranklist, gt_item):
    return int(gt_item in ranklist)


def ndcg(ranklist, gt_item):
    for idx, item in enumerate(ranklist):
        if item == gt_item:
            return math.log(2) / math.log(idx + 2)
    return 0.0


def load_processed_data(processed_dir):
    processed_dir = Path(processed_dir)

    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    user_mapping = pd.read_csv(processed_dir / "user_mapping.csv")
    item_mapping = pd.read_csv(processed_dir / "item_mapping.csv")

    num_users = user_mapping["user_idx"].nunique()
    num_items = item_mapping["item_idx"].nunique()

    return train, val, test, num_users, num_items


def build_seen_items(train, val, test):
    all_interactions = pd.concat([train, val, test], ignore_index=True)

    seen = {}
    for row in all_interactions.itertuples(index=False):
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


def load_trained_model(model_name, num_users, num_items, runs_dir, device):
    model_dir = Path(runs_dir) / model_name
    summary_path = model_dir / "summary.json"
    weights_path = model_dir / "best_model.pt"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    params = summary["params"]

    layers = params.get("layers", (64, 32, 16, 8))
    if isinstance(layers, str):
        layers = tuple(int(x.strip()) for x in layers.split(","))
    elif isinstance(layers, list):
        layers = tuple(int(x) for x in layers)
    else:
        layers = tuple(layers)

    dropout = params.get("dropout", 0.0)
    if dropout is None:
        dropout = 0.0

    embedding_dim = params.get("embedding_dim", 8)

    model = get_model(
        model_name=model_name,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=int(embedding_dim),
        layers=layers,
        dropout=float(dropout),
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model


def score_candidates(model, user, candidate_items, device):
    users = torch.full((len(candidate_items),), user, dtype=torch.long, device=device)
    items = torch.tensor(candidate_items, dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(users, items).detach().cpu().numpy()

    return scores


def evaluate_model_ranking(
    model,
    test,
    num_items,
    seen_items,
    k=10,
    num_negatives=99,
    seed=42,
    max_eval_users=None,
    device="cpu",
):
    rng = np.random.default_rng(seed)

    test_positives = test[test["label"] == 1][["user_idx", "item_idx"]].copy()

    if max_eval_users is not None:
        test_positives = test_positives.head(max_eval_users)

    hits = []
    ndcgs = []

    for row in test_positives.itertuples(index=False):
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

        scores = score_candidates(
            model=model,
            user=user,
            candidate_items=candidate_items,
            device=device,
        )

        item_score = dict(zip(candidate_items, scores))

        ranklist = sorted(candidate_items, key=lambda item: item_score[item], reverse=True)[:k]

        hits.append(hit_ratio(ranklist, gt_item))
        ndcgs.append(ndcg(ranklist, gt_item))

    return {
        f"HR@{k}": float(np.mean(hits)) if hits else None,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        "num_eval_positives": int(len(test_positives)),
        "num_negatives": int(num_negatives),
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train, val, test, num_users, num_items = load_processed_data(args.processed_dir)
    seen_items = build_seen_items(train, val, test)

    print("Num users:", num_users)
    print("Num items:", num_items)
    print("Test positives:", int((test["label"] == 1).sum()))

    if args.model == "all":
        model_names = ["gmf", "mlp", "neumf"]
    else:
        model_names = [args.model]

    rows = []

    for model_name in model_names:
        print("\n" + "=" * 80)
        print(f"Ranking evaluation: {model_name.upper()}")
        print("=" * 80)

        model = load_trained_model(
            model_name=model_name,
            num_users=num_users,
            num_items=num_items,
            runs_dir=args.runs_dir,
            device=device,
        )

        metrics = evaluate_model_ranking(
            model=model,
            test=test,
            num_items=num_items,
            seen_items=seen_items,
            k=args.k,
            num_negatives=args.num_negatives,
            seed=args.seed,
            max_eval_users=args.max_eval_users,
            device=device,
        )

        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

        print(json.dumps(row, indent=4))

    out_path = Path(args.runs_dir) / f"ranking_comparison_k{args.k}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("\nSaved ranking comparison to:", out_path)


if __name__ == "__main__":
    main()
