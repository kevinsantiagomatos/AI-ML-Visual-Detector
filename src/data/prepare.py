"""Prepare processed train/val/test splits with symlinks and metadata.

Usage:
  python -m src.data.prepare --dataset flower_photos
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


VALID_SPLITS = {"train", "val", "test"}


def stratified_split(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be between 0 and 1")
    remaining_ratio = 1 - val_ratio - test_ratio
    if remaining_ratio <= 0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    train_df, temp_df = train_test_split(df, test_size=val_ratio + test_ratio, stratify=df["class"], random_state=seed)
    relative_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
    val_df, test_df = train_test_split(temp_df, test_size=relative_test, stratify=temp_df["class"], random_state=seed)

    splits = pd.Series(index=df.index, dtype="object")
    splits.loc[train_df.index] = "train"
    splits.loc[val_df.index] = "val"
    splits.loc[test_df.index] = "test"
    return splits


def ensure_symlink(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    os.symlink(src, dest)


def compute_brightness(path: Path) -> float:
    with Image.open(path) as img:
        gray = img.convert("L")
        return float(np.mean(gray))


def build_symlink_tree(df: pd.DataFrame, processed_root: Path):
    records = []
    for _, row in df.iterrows():
        src = Path(row["path"])
        split = row["split"]
        cls = row["class"]
        dest = processed_root / split / cls / src.name
        ensure_symlink(src, dest)
        records.append({"path": str(dest.resolve()), **row.to_dict()})
    return pd.DataFrame(records)


def compute_stats(df: pd.DataFrame) -> Dict:
    class_counts = Counter(df["class"])
    split_counts = Counter(df["split"])
    brightness_samples = [compute_brightness(Path(p)) for p in df.sample(min(len(df), 200), random_state=42)["path"]]
    return {
        "num_rows": int(len(df)),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "brightness_mean": float(np.mean(brightness_samples)) if brightness_samples else 0.0,
        "brightness_std": float(np.std(brightness_samples)) if brightness_samples else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare processed splits and metadata")
    parser.add_argument("--dataset", required=True, help="Dataset name (folder in data/raw)")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = args.raw_dir / args.dataset / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run ingest first.")

    df = pd.read_csv(manifest_path)
    if not set(df["split"].unique()) & VALID_SPLITS:
        df["split"] = stratified_split(df, args.val_ratio, args.test_ratio, args.seed)

    processed_root = args.out_dir / args.dataset
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_df = build_symlink_tree(df, processed_root)
    processed_manifest = processed_root / "manifest.csv"
    processed_df.to_csv(processed_manifest, index=False)

    stats = compute_stats(processed_df)
    metadata = {
        "dataset": args.dataset,
        "source_manifest": str(manifest_path),
        "processed_root": str(processed_root.resolve()),
        "class_names": sorted(processed_df["class"].unique().tolist()),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "stats": stats,
    }
    metadata_path = processed_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    reference_stats_path = processed_root / "reference_stats.json"
    if not reference_stats_path.exists():
        reference_stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Processed splits written to {processed_root}")
    print(f"Class counts: {stats['class_counts']}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
