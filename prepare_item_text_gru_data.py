
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_jsonl", required=True)
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_vocab_size", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--max_reviews_per_item", type=int, default=5)
    p.add_argument("--max_chars_per_review", type=int, default=500)
    return p.parse_args()


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


def tokenize(text):
    if text is None:
        return []
    return TOKEN_RE.findall(str(text).lower())


def read_raw_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def get_text(record):
    parts = []
    for key in ["text", "review_text", "reviewText", "title", "description"]:
        val = record.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = " ".join(str(x) for x in val)
        parts.append(str(val))
    return " ".join(parts).strip()


def main():
    args = parse_args()
    processed = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_mapping = pd.read_csv(processed / "user_mapping.csv")
    item_mapping = pd.read_csv(processed / "item_mapping.csv")
    train = pd.read_csv(processed / "train.csv")

    user_raw_col = find_col(user_mapping, ["user_id", "raw_user_id", "reviewerID", "reviewer_id"])
    item_raw_col = find_col(item_mapping, ["item_id", "product_id", "parent_asin", "asin", "raw_item_id"])

    user_to_idx = dict(zip(user_mapping[user_raw_col].astype(str), user_mapping["user_idx"].astype(int)))
    item_to_idx = dict(zip(item_mapping[item_raw_col].astype(str), item_mapping["item_idx"].astype(int)))
    train_pairs = set(zip(train["user_idx"].astype(int), train["item_idx"].astype(int)))

    num_items = item_mapping["item_idx"].nunique()
    item_texts = defaultdict(list)

    raw_count = 0
    used_count = 0

    for rec in read_raw_jsonl(args.raw_jsonl):
        raw_count += 1

        raw_user = rec.get("user_id", rec.get("reviewerID", rec.get("reviewer_id")))
        raw_item = rec.get("parent_asin", rec.get("asin", rec.get("product_id", rec.get("item_id"))))

        if raw_user is None or raw_item is None:
            continue

        user_idx = user_to_idx.get(str(raw_user))
        item_idx = item_to_idx.get(str(raw_item))

        if user_idx is None or item_idx is None:
            continue

        if (user_idx, item_idx) not in train_pairs:
            continue

        if len(item_texts[item_idx]) >= args.max_reviews_per_item:
            continue

        text = get_text(rec)
        if not text:
            continue

        item_texts[item_idx].append(text[: args.max_chars_per_review])
        used_count += 1

    counter = Counter()
    for texts in item_texts.values():
        for text in texts:
            counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(max(args.max_vocab_size - 2, 0)):
        if token not in vocab:
            vocab[token] = len(vocab)

    item_tokens = np.zeros((num_items, args.max_len), dtype=np.int64)
    for item_idx in range(num_items):
        text = " ".join(item_texts.get(item_idx, []))
        ids = [vocab.get(tok, 1) for tok in tokenize(text)[: args.max_len]]
        if ids:
            item_tokens[item_idx, : len(ids)] = ids

    np.save(out_dir / "item_tokens.npy", item_tokens)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    summary = {
        "raw_jsonl": str(args.raw_jsonl),
        "processed_dir": str(args.processed_dir),
        "num_items": int(num_items),
        "raw_records_seen": int(raw_count),
        "train_text_records_used": int(used_count),
        "items_with_text": int((item_tokens.sum(axis=1) > 0).sum()),
        "vocab_size": int(len(vocab)),
        "max_len": int(args.max_len),
        "max_reviews_per_item": int(args.max_reviews_per_item),
        "safe_text_source": "training reviews only",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
