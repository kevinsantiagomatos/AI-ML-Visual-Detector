"""Ingest public image datasets from config and materialize manifests.

Usage:
  python -m src.data.ingest --config data/sources/sources.yaml --datasets flower_photos hymenoptera

Outputs:
  data/raw/<dataset>/                             # extracted files
  data/raw/<dataset>/manifest.csv                 # file-level manifest
"""
import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
import tqdm
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, checksum: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=f"downloading {dest.name}") as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    if checksum:
        actual = sha256_file(dest)
        if actual != checksum:
            raise ValueError(f"checksum mismatch for {dest.name}: expected {checksum}, got {actual}")
    return dest


def extract_archive(archive_path: Path, target_dir: Path, archive_type: str):
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_type in {"tgz", "tar.gz", "tar"}:
        mode = "r:gz" if archive_type != "tar" else "r"
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(target_dir)
    elif archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive_type {archive_type}")


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def normalize_split(parts: List[str]) -> str:
    for p in parts:
        if p.lower() in {"train", "training"}:
            return "train"
        if p.lower() in {"val", "valid", "validation"}:
            return "val"
        if p.lower() in {"test", "testing"}:
            return "test"
    return "unsplit"


def build_manifest(extracted_root: Path) -> pd.DataFrame:
    records = []
    for img_path in iter_images(extracted_root):
        parts = img_path.relative_to(extracted_root).parts
        cls = parts[-2] if len(parts) >= 2 else "unknown"
        split = normalize_split(parts)
        records.append(
            {
                "path": str(img_path.resolve()),
                "rel_path": str(img_path.relative_to(extracted_root)),
                "split": split,
                "class": cls,
                "sha256": sha256_file(img_path),
                "bytes": img_path.stat().st_size,
            }
        )
    return pd.DataFrame(records)


def load_sources(config_path: Path) -> Dict[str, Dict]:
    cfg = yaml.safe_load(config_path.read_text())
    return cfg.get("datasets", {})


def ingest_dataset(name: str, cfg: Dict, raw_dir: Path):
    print(f"\n=== Ingesting {name} ===")
    url = cfg["url"]
    archive_type = cfg.get("archive_type", "tgz")
    checksum = cfg.get("checksum", "")
    download_path = raw_dir / f"{name}.{archive_type.replace('.', '')}"
    download_file(url, download_path, checksum)

    extract_dir = raw_dir / name
    extract_archive(download_path, extract_dir, archive_type)

    manifest = build_manifest(extract_dir)
    manifest_path = extract_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest with {len(manifest)} rows to {manifest_path}")
    print(manifest.groupby(["split", "class"]).size())


def main():
    parser = argparse.ArgumentParser(description="Ingest public image datasets")
    parser.add_argument("--config", type=Path, default=Path("data/sources/sources.yaml"))
    parser.add_argument("--datasets", nargs="*", help="Datasets to ingest (default: all in config)")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    sources = load_sources(args.config)
    targets = args.datasets or list(sources.keys())
    missing = [d for d in targets if d not in sources]
    if missing:
        raise SystemExit(f"Datasets not found in config: {missing}")

    for name in targets:
        ingest_dataset(name, sources[name], args.raw_dir)


if __name__ == "__main__":
    main()
