"""Prefect flow that chains ingest -> prepare -> quality.

Run locally:
  prefect deployment run data_pipeline --params '{"datasets": ["flower_photos"]}'
"""
import subprocess
from typing import List, Optional
from pathlib import Path
import sys

from prefect import flow, task


def _run(cmd: List[str]):
    print("Running", " ".join(cmd))
    subprocess.run(cmd, check=True)


@task
def ingest_task(datasets: Optional[List[str]], config: str, raw_dir: str):
    cmd = [sys.executable, "-m", "src.data.ingest", "--config", config, "--raw_dir", raw_dir]
    if datasets:
        cmd.append("--datasets")
        cmd.extend(datasets)
    _run(cmd)


@task
def prepare_task(dataset: str, raw_dir: str, out_dir: str, val_ratio: float, test_ratio: float):
    cmd = [
        sys.executable,
        "-m",
        "src.data.prepare",
        "--dataset",
        dataset,
        "--raw_dir",
        raw_dir,
        "--out_dir",
        out_dir,
        "--val_ratio",
        str(val_ratio),
        "--test_ratio",
        str(test_ratio),
    ]
    _run(cmd)


@task
def quality_task(dataset: str, processed_dir: str, reports_dir: str, reference: Optional[str]):
    cmd = [
        sys.executable,
        "-m",
        "src.data.quality",
        "--dataset",
        dataset,
        "--processed_dir",
        processed_dir,
        "--reports_dir",
        reports_dir,
    ]
    if reference:
        cmd.extend(["--reference", reference])
    _run(cmd)


@flow(name="data_pipeline")
def data_pipeline(
    datasets: Optional[List[str]] = None,
    config: str = "data/sources/sources.yaml",
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    reports_dir: str = "reports/ge",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    ingest_task(datasets, config, raw_dir)
    targets = datasets
    if targets is None:
        import yaml

        cfg = yaml.safe_load(Path(config).read_text())
        targets = list(cfg.get("datasets", {}).keys())

    for ds in targets:
        prepare_task(ds, raw_dir, processed_dir, val_ratio, test_ratio)
        reference_path = str(Path(processed_dir) / ds / "reference_stats.json") if Path(processed_dir, ds, "reference_stats.json").exists() else None
        quality_task(ds, processed_dir, reports_dir, reference_path)


if __name__ == "__main__":
    data_pipeline()
