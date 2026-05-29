import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import get_model


class BPRDataset(Dataset):
    def __init__(self, positive_df, seen_items, num_items, seed=42):
        self.users = positive_df["user_idx"].astype(int).values
        self.pos_items = positive_df["item_idx"].astype(int).values
        self.seen_items = seen_items
        self.num_items = int(num_items)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos_item = int(self.pos_items[idx])
        user_seen = self.seen_items.get(user, set())

        while True:
            neg_item = int(self.rng.integers(0, self.num_items))
            if neg_item != pos_item and neg_item not in user_seen:
                break

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="all", choices=["gmf", "mlp", "neumf", "all"])
    parser.add_argument("--processed_dir", type=str, default="Data/Processed")
    parser.add_argument("--output_dir", type=str, default="Runs/BPR")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--layers", type=str, default="64,32,16,8")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--selection_metric", type=str, default="hr", choices=["hr", "ndcg", "loss"])
    parser.add_argument("--ranking_k", type=int, default=10)
    parser.add_argument("--val_num_negatives", type=int, default=99)
    parser.add_argument("--max_val_positives", type=int, default=5000,
                        help="Limit validation positives for faster HR/NDCG each epoch. Use 0 for all positives.")

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_layers(layers):
    if isinstance(layers, str):
        return tuple(int(x.strip()) for x in layers.split(","))
    if isinstance(layers, list):
        return tuple(int(x) for x in layers)
    return tuple(layers)


def load_data(processed_dir):
    p = Path(processed_dir)

    train = pd.read_csv(p / "train.csv")
    val = pd.read_csv(p / "val.csv")
    test = pd.read_csv(p / "test.csv")

    user_mapping = pd.read_csv(p / "user_mapping.csv")
    item_mapping = pd.read_csv(p / "item_mapping.csv")

    num_users = user_mapping["user_idx"].nunique()
    num_items = item_mapping["item_idx"].nunique()

    return train, val, test, num_users, num_items


def build_seen_items(*dfs):
    seen = {}
    for df in dfs:
        for row in df[["user_idx", "item_idx"]].itertuples(index=False):
            user = int(row.user_idx)
            item = int(row.item_idx)
            seen.setdefault(user, set()).add(item)
    return seen


def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []

    for users, pos_items, neg_items in loader:
        users = users.to(device, non_blocking=True)
        pos_items = pos_items.to(device, non_blocking=True)
        neg_items = neg_items.to(device, non_blocking=True)

        optimizer.zero_grad()

        pos_scores = model(users, pos_items)
        neg_scores = model(users, neg_items)

        loss = bpr_loss(pos_scores, neg_scores)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def evaluate_ranking_validation(
    model,
    val_df,
    num_items,
    seen_items,
    device,
    k=10,
    num_negatives=99,
    max_val_positives=5000,
    seed=42,
):
    model.eval()
    rng = np.random.default_rng(seed)

    positives = val_df[val_df["label"] == 1][["user_idx", "item_idx"]].copy()

    if max_val_positives is not None and max_val_positives > 0 and len(positives) > max_val_positives:
        positives = positives.sample(n=max_val_positives, random_state=seed).reset_index(drop=True)

    hits = []
    ndcgs = []

    with torch.no_grad():
        for row in positives.itertuples(index=False):
            user = int(row.user_idx)
            pos_item = int(row.item_idx)

            user_seen = seen_items.get(user, set())
            negatives = []
            attempts = 0
            max_attempts = num_negatives * 100

            while len(negatives) < num_negatives and attempts < max_attempts:
                neg_item = int(rng.integers(0, num_items))
                attempts += 1

                if neg_item == pos_item:
                    continue

                if neg_item in user_seen:
                    continue

                negatives.append(neg_item)

            while len(negatives) < num_negatives:
                neg_item = int(rng.integers(0, num_items))
                if neg_item != pos_item:
                    negatives.append(neg_item)

            candidate_items = [pos_item] + negatives
            candidate_users = [user] * len(candidate_items)

            users_tensor = torch.tensor(candidate_users, dtype=torch.long, device=device)
            items_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)

            scores = model(users_tensor, items_tensor).detach().cpu().numpy()
            ranking = np.argsort(-scores)

            pos_rank = int(np.where(ranking == 0)[0][0])

            if pos_rank < k:
                hits.append(1.0)
                ndcgs.append(1.0 / np.log2(pos_rank + 2))
            else:
                hits.append(0.0)
                ndcgs.append(0.0)

    if len(hits) == 0:
        return {
            f"HR@{k}": None,
            f"NDCG@{k}": None,
            "num_eval_positives": 0,
            "num_negatives": num_negatives,
        }

    return {
        f"HR@{k}": float(np.mean(hits)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "num_eval_positives": int(len(hits)),
        "num_negatives": int(num_negatives),
    }


def get_selection_value(args, train_loss, ranking_metrics):
    if args.selection_metric == "hr":
        return ranking_metrics[f"HR@{args.ranking_k}"]
    if args.selection_metric == "ndcg":
        return ranking_metrics[f"NDCG@{args.ranking_k}"]
    if args.selection_metric == "loss":
        return -train_loss
    raise ValueError(args.selection_metric)


def format_float(x):
    if x is None:
        return "None"
    return f"{x:.4f}"


def train_model(model_name, train, val, test, num_users, num_items, args, device):
    layers = parse_layers(args.layers)

    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_items = build_seen_items(train, val, test)

    positive_train = train[train["label"] == 1].copy()

    loader = DataLoader(
        BPRDataset(positive_train, seen_items, num_items, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = get_model(
        model_name=model_name,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        layers=layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    epochs_without_improvement = 0
    stopped_early = False
    history = []

    print("\n" + "=" * 80)
    print(f"Training BPR model: {model_name.upper()}")
    print("=" * 80)
    print(
        f"dropout={args.dropout}, weight_decay={args.weight_decay}, "
        f"patience={args.patience}, selection_metric={args.selection_metric}"
    )

    # Epoch 0: random initialized model
    val_ranking = evaluate_ranking_validation(
        model=model,
        val_df=val,
        num_items=num_items,
        seen_items=seen_items,
        device=device,
        k=args.ranking_k,
        num_negatives=args.val_num_negatives,
        max_val_positives=args.max_val_positives,
        seed=args.seed,
    )

    current_metric = get_selection_value(args, 0.0, val_ranking)

    history.append({
        "epoch": 0,
        "train_bpr_loss": None,
        f"val_HR@{args.ranking_k}": val_ranking[f"HR@{args.ranking_k}"],
        f"val_NDCG@{args.ranking_k}": val_ranking[f"NDCG@{args.ranking_k}"],
        "val_num_eval_positives": val_ranking["num_eval_positives"],
        "selection_metric": args.selection_metric,
        "selection_value": current_metric,
    })

    print(
        f"Epoch 000 | train_bpr_loss=None | "
        f"val_HR@{args.ranking_k}={format_float(val_ranking[f'HR@{args.ranking_k}'])} | "
        f"val_NDCG@{args.ranking_k}={format_float(val_ranking[f'NDCG@{args.ranking_k}'])} | "
        f"select={format_float(current_metric)}"
    )

    if current_metric is not None:
        best_metric = current_metric
        best_epoch = 0
        torch.save(model.state_dict(), best_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device)

        val_ranking = evaluate_ranking_validation(
            model=model,
            val_df=val,
            num_items=num_items,
            seen_items=seen_items,
            device=device,
            k=args.ranking_k,
            num_negatives=args.val_num_negatives,
            max_val_positives=args.max_val_positives,
            seed=args.seed + epoch,
        )

        current_metric = get_selection_value(args, train_loss, val_ranking)

        history.append({
            "epoch": epoch,
            "train_bpr_loss": train_loss,
            f"val_HR@{args.ranking_k}": val_ranking[f"HR@{args.ranking_k}"],
            f"val_NDCG@{args.ranking_k}": val_ranking[f"NDCG@{args.ranking_k}"],
            "val_num_eval_positives": val_ranking["num_eval_positives"],
            "selection_metric": args.selection_metric,
            "selection_value": current_metric,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_bpr_loss={train_loss:.4f} | "
            f"val_HR@{args.ranking_k}={format_float(val_ranking[f'HR@{args.ranking_k}'])} | "
            f"val_NDCG@{args.ranking_k}={format_float(val_ranking[f'NDCG@{args.ranking_k}'])} | "
            f"select={format_float(current_metric)}"
        )

        if current_metric is not None and current_metric > best_metric + args.min_delta:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best {args.selection_metric}: {best_metric:.4f}"
            )
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    summary = {
        "model": model_name,
        "loss": "BPR",
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_value": best_metric,
        "stopped_early": stopped_early,
        "epochs_ran": len(history) - 1,
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "embedding_dim": args.embedding_dim,
            "layers": layers,
            "dropout": args.dropout,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "selection_metric": args.selection_metric,
            "ranking_k": args.ranking_k,
            "val_num_negatives": args.val_num_negatives,
            "max_val_positives": args.max_val_positives,
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train, val, test, num_users, num_items = load_data(args.processed_dir)

    print("Train shape:", train.shape)
    print("Val shape:", val.shape)
    print("Test shape:", test.shape)
    print("Num users:", num_users)
    print("Num items:", num_items)

    if args.model == "all":
        model_names = ["gmf", "mlp", "neumf"]
    else:
        model_names = [args.model]

    summaries = []

    for model_name in model_names:
        summaries.append(
            train_model(model_name, train, val, test, num_users, num_items, args, device)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in summaries:
        rows.append({
            "model": s["model"],
            "loss": s["loss"],
            "selection_metric": s["selection_metric"],
            "best_epoch": s["best_epoch"],
            "best_selection_value": s["best_selection_value"],
            "epochs_ran": s["epochs_ran"],
            "stopped_early": s["stopped_early"],
            "dropout": s["params"]["dropout"],
            "weight_decay": s["params"]["weight_decay"],
            "patience": s["params"]["patience"],
            "lr": s["params"]["lr"],
            "ranking_k": s["params"]["ranking_k"],
            "val_num_negatives": s["params"]["val_num_negatives"],
        })

    pd.DataFrame(rows).to_csv(output_dir / "comparison_bpr.csv", index=False)

    print("\nSaved comparison to:", output_dir / "comparison_bpr.csv")


if __name__ == "__main__":
    main()
