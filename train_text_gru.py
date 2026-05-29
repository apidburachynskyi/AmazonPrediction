import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import Dataset, DataLoader

from models_text import NeuMFWithGRU


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--processed_dir", type=str, default="Data/ProcessedText")
    parser.add_argument("--output_dir", type=str, default="Runs/TextGRU")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--gmf_embedding_dim", type=int, default=8)
    parser.add_argument("--mlp_layers", type=str, default="64,32,16,8")
    parser.add_argument("--word_embedding_dim", type=int, default=64)
    parser.add_argument("--gru_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--max_vocab_size", type=int, default=20000)
    parser.add_argument("--max_len", type=int, default=50)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-Z0-9']+", text)


def build_vocab(texts, max_vocab_size):
    counter = Counter()

    for text in texts:
        counter.update(simple_tokenize(text))

    most_common = counter.most_common(max_vocab_size - 2)

    word_to_idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, _ in most_common:
        word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def encode_text(text, word_to_idx, max_len):
    tokens = simple_tokenize(text)
    ids = [word_to_idx.get(tok, word_to_idx[UNK_TOKEN]) for tok in tokens[:max_len]]

    if len(ids) < max_len:
        ids += [word_to_idx[PAD_TOKEN]] * (max_len - len(ids))

    return ids


class TextInteractionDataset(Dataset):
    def __init__(self, df, word_to_idx, max_len):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

        encoded = [
            encode_text(text, word_to_idx, max_len)
            for text in df["review_text"].fillna("").astype(str).values
        ]
        self.tokens = torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.tokens[idx], self.labels[idx]


def load_data(processed_dir):
    processed_dir = Path(processed_dir)

    train = pd.read_csv(processed_dir / "train_text.csv")
    val = pd.read_csv(processed_dir / "val_text.csv")
    test = pd.read_csv(processed_dir / "test_text.csv")

    user_mapping = pd.read_csv(processed_dir / "user_mapping.csv")
    item_mapping = pd.read_csv(processed_dir / "item_mapping.csv")

    num_users = user_mapping["user_idx"].nunique()
    num_items = item_mapping["item_idx"].nunique()

    return train, val, test, num_users, num_items


def make_loaders(train, val, test, word_to_idx, max_len, batch_size):
    train_loader = DataLoader(
        TextInteractionDataset(train, word_to_idx, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        TextInteractionDataset(val, word_to_idx, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        TextInteractionDataset(test, word_to_idx, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    losses = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for users, items, tokens, labels in loader:
            users = users.to(device)
            items = items.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device)

            probs = model(users, items, tokens)
            loss = criterion(probs, labels)

            losses.append(loss.item())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = None

    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None

    return metrics


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


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train, val, test, num_users, num_items = load_data(args.processed_dir)

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)
    print("Num users:", num_users)
    print("Num items:", num_items)

    print("Bulding")
    word_to_idx = build_vocab(train["review_text"].fillna("").astype(str).values, args.max_vocab_size)
    vocab_size = len(word_to_idx)
    print("Vocab size:", vocab_size)

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f)

    train_loader, val_loader, test_loader = make_loaders(
        train=train,
        val=val,
        test=test,
        word_to_idx=word_to_idx,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    mlp_layers = tuple(int(x) for x in args.mlp_layers.split(","))

    model = NeuMFWithGRU(
        num_users=num_users,
        num_items=num_items,
        vocab_size=vocab_size,
        gmf_embedding_dim=args.gmf_embedding_dim,
        mlp_layers=mlp_layers,
        word_embedding_dim=args.word_embedding_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        dropout=args.dropout,
        padding_idx=0,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_auc = -1
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_log_loss": val_metrics["log_loss"],
        }
        history.append(row)

        val_auc_print = "None" if val_metrics["auc"] is None else f"{val_metrics['auc']:.4f}"

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_auc_print}"
        )

        current_auc = val_metrics["auc"]
        if current_auc is not None and current_auc > best_val_auc:
            best_val_auc = current_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"stopping at epoch {epoch} Best epoch: {best_epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, args.threshold)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    summary = {
        "model": "NeuMFWithGRU",
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "params": vars(args),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=4))
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
