
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import Dataset, DataLoader

from models_text import NeuMFWithGRU


class TextInteractionDataset(Dataset):
    def __init__(self, df, item_tokens):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)
        self.item_tokens = item_tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = int(self.items[idx])
        tokens = torch.tensor(self.item_tokens[item], dtype=torch.long)
        return self.users[idx], self.items[idx], tokens, self.labels[idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--text_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight_decay", type=float, default=0.000001)
    p.add_argument("--layers", type=str, default="64,32,16,8")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gmf_embedding_dim", type=int, default=8)
    p.add_argument("--word_embedding_dim", type=int, default=64)
    p.add_argument("--gru_hidden_dim", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--selection_metric", choices=["hr", "ndcg", "auc", "loss"], default="hr")
    p.add_argument("--ranking_k", type=int, default=10)
    p.add_argument("--val_num_negatives", type=int, default=99)
    p.add_argument("--max_val_positives", type=int, default=5000)
    p.add_argument("--max_test_positives", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_layers(s):
    return tuple(int(x.strip()) for x in s.split(","))


def build_seen_items(*dfs):
    seen = {}
    for df in dfs:
        for r in df[["user_idx", "item_idx"]].itertuples(index=False):
            seen.setdefault(int(r.user_idx), set()).add(int(r.item_idx))
    return seen


def make_loader(df, item_tokens, batch_size, shuffle):
    return DataLoader(
        TextInteractionDataset(df, item_tokens),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate_classification(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_prob = [], [], []

    with torch.no_grad():
        for users, items, tokens, labels in loader:
            users = users.to(device)
            items = items.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device)
            probs = model(users, items, tokens)
            loss = criterion(probs, labels)
            losses.append(loss.item())
            y_true.extend(labels.detach().cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    out = {"loss": float(np.mean(losses)), "accuracy": float(accuracy_score(y_true, y_pred))}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = None
    try:
        out["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        out["log_loss"] = None
    return out


def evaluate_ranking(model, eval_df, item_tokens, num_items, seen_items, device, k, num_negatives, max_eval_positives, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    positives = eval_df[eval_df["label"] == 1][["user_idx", "item_idx"]].copy()

    if max_eval_positives and len(positives) > max_eval_positives:
        positives = positives.sample(n=max_eval_positives, random_state=seed).reset_index(drop=True)

    hits, ndcgs = [], []

    with torch.no_grad():
        for r in positives.itertuples(index=False):
            user = int(r.user_idx)
            pos_item = int(r.item_idx)
            seen = seen_items.get(user, set())

            negatives = []
            while len(negatives) < num_negatives:
                item = int(rng.integers(0, num_items))
                if item == pos_item or item in seen:
                    continue
                negatives.append(item)

            candidates = [pos_item] + negatives
            users = torch.full((len(candidates),), user, dtype=torch.long, device=device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            tokens_np = item_tokens[np.asarray(candidates, dtype=np.int64)]
            tokens = torch.tensor(tokens_np, dtype=torch.long, device=device)

            scores = model(users, items, tokens).detach().cpu().numpy()
            ranking = np.argsort(-scores)
            pos_rank = int(np.where(ranking == 0)[0][0])

            if pos_rank < k:
                hits.append(1.0)
                ndcgs.append(1.0 / np.log2(pos_rank + 2))
            else:
                hits.append(0.0)
                ndcgs.append(0.0)

    return {
        f"HR@{k}": float(np.mean(hits)) if hits else None,
        f"NDCG@{k}": float(np.mean(ndcgs)) if hits else None,
        "num_eval_positives": int(len(hits)),
        "num_negatives": int(num_negatives),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for users, items, tokens, labels in loader:
        users = users.to(device)
        items = items.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        probs = model(users, items, tokens)
        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def selection_value(args, class_metrics, rank_metrics):
    if args.selection_metric == "hr":
        return rank_metrics[f"HR@{args.ranking_k}"]
    if args.selection_metric == "ndcg":
        return rank_metrics[f"NDCG@{args.ranking_k}"]
    if args.selection_metric == "auc":
        return class_metrics["auc"]
    if args.selection_metric == "loss":
        return -class_metrics["loss"]
    raise ValueError(args.selection_metric)


def fmt(x):
    if x is None:
        return "None"
    return f"{x:.4f}"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processed = Path(args.processed_dir)
    text_dir = Path(args.text_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed / "train_neg_sampled.csv")
    train_original = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "val.csv")
    test = pd.read_csv(processed / "test.csv")
    user_mapping = pd.read_csv(processed / "user_mapping.csv")
    item_mapping = pd.read_csv(processed / "item_mapping.csv")

    item_tokens = np.load(text_dir / "item_tokens.npy")
    with open(text_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    num_users = user_mapping["user_idx"].nunique()
    num_items = item_mapping["item_idx"].nunique()
    vocab_size = len(vocab)

    print("Train sampled shape:", train.shape)
    print("Val shape:", val.shape)
    print("Test shape:", test.shape)
    print("Num users:", num_users)
    print("Num items:", num_items)
    print("Vocab size:", vocab_size)
    print("Items with text:", int((item_tokens.sum(axis=1) > 0).sum()))

    seen_items = build_seen_items(train_original, val, test)
    train_loader = make_loader(train, item_tokens, args.batch_size, True)
    val_loader = make_loader(val, item_tokens, args.batch_size, False)
    test_loader = make_loader(test, item_tokens, args.batch_size, False)

    layers = parse_layers(args.layers)
    model = NeuMFWithGRU(
        num_users=num_users,
        num_items=num_items,
        vocab_size=vocab_size,
        gmf_embedding_dim=args.gmf_embedding_dim,
        mlp_layers=layers,
        word_embedding_dim=args.word_embedding_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        dropout=args.dropout,
        padding_idx=0,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_metric = -float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    epochs_without_improvement = 0
    stopped_early = False

    print("=" * 80)
    print("Training NeuMF + GRU item-text recommender")
    print("=" * 80)

    for epoch in range(0, args.epochs + 1):
        if epoch == 0:
            train_loss = None
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_class = evaluate_classification(model, val_loader, criterion, device)
        val_rank = evaluate_ranking(
            model, val, item_tokens, num_items, seen_items, device,
            args.ranking_k, args.val_num_negatives, args.max_val_positives, args.seed + epoch,
        )
        current = selection_value(args, val_class, val_rank)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_class["loss"],
            "val_auc": val_class["auc"],
            f"val_HR@{args.ranking_k}": val_rank[f"HR@{args.ranking_k}"],
            f"val_NDCG@{args.ranking_k}": val_rank[f"NDCG@{args.ranking_k}"],
            "selection_value": current,
        })

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss if train_loss is not None else 'None'} | "
            f"val_loss={fmt(val_class['loss'])} | val_auc={fmt(val_class['auc'])} | "
            f"val_HR@{args.ranking_k}={fmt(val_rank[f'HR@{args.ranking_k}'])} | "
            f"val_NDCG@{args.ranking_k}={fmt(val_rank[f'NDCG@{args.ranking_k}'])} | select={fmt(current)}"
        )

        if current is not None and current > best_metric + args.min_delta:
            best_metric = current
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            if epoch > 0:
                epochs_without_improvement += 1

        if epoch > 0 and args.patience > 0 and epochs_without_improvement >= args.patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best {args.selection_metric}: {best_metric:.4f}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))

    test_class = evaluate_classification(model, test_loader, criterion, device)
    test_rank = evaluate_ranking(
        model, test, item_tokens, num_items, seen_items, device,
        args.ranking_k, args.val_num_negatives, args.max_test_positives, args.seed + 999,
    )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    summary = {
        "model": "NeuMF+GRU",
        "loss": "BCE",
        "uses_text": True,
        "text_source": "item-level text from training reviews only",
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_value": best_metric,
        "stopped_early": stopped_early,
        "epochs_ran": int(max(0, len(history) - 1)),
        "test_metrics": test_class,
        "test_ranking": test_rank,
        "params": vars(args),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    ranking_row = {
        "model": "neumf_gru",
        f"HR@{args.ranking_k}": test_rank[f"HR@{args.ranking_k}"],
        f"NDCG@{args.ranking_k}": test_rank[f"NDCG@{args.ranking_k}"],
        "num_eval_positives": test_rank["num_eval_positives"],
        "num_negatives": test_rank["num_negatives"],
    }
    pd.DataFrame([ranking_row]).to_csv(output_dir / f"ranking_comparison_k{args.ranking_k}.csv", index=False)

    print("\nTest classification:")
    print(json.dumps(test_class, indent=4))
    print("\nTest ranking:")
    print(json.dumps(test_rank, indent=4))
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
