"""
download_data.py

Download one category from Amazon Reviews 2023 into Data/Raw.

Example:
    python download_data.py --category All_Beauty

List available categories:
    python download_data.py --list
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REPO_TYPE = "dataset"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        default="All_Beauty",
        help="Amazon category to download, e.g. All_Beauty, Amazon_Fashion",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Data/Raw",
        help="Folder where the jsonl file will be saved.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available review categories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite file if it already exists.",
    )
    return parser.parse_args()


def list_categories():
    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)

    categories = []
    for f in files:
        if f.startswith("raw/review_categories/") and f.endswith(".jsonl"):
            name = Path(f).stem
            categories.append(name)

    categories = sorted(categories)

    print("Available categories:")
    for c in categories:
        print("-", c)


def download_category(category: str, out_dir: str, force: bool = False):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"raw/review_categories/{category}.jsonl"
    final_path = out_path / f"{category}.jsonl"

    if final_path.exists() and not force:
        print(f"File already exists: {final_path}")
        print("Use --force if you want to overwrite it.")
        return final_path

    print(f"Downloading category: {category}")
    print(f"Remote file: {filename}")

    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
    )

    shutil.copy2(cached_path, final_path)

    print(f"Saved to: {final_path}")
    return final_path


def main():
    args = parse_args()

    if args.list:
        list_categories()
        return

    download_category(
        category=args.category,
        out_dir=args.out_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()