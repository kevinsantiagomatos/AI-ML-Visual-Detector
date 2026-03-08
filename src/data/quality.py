"""Profile processed data with Great Expectations and basic drift checks.

Usage:
  python -m src.data.quality --dataset flower_photos
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from great_expectations.dataset import PandasDataset
from great_expectations.render.renderer import ValidationResultsPageRenderer
from great_expectations.render.view import DefaultJinjaPageView
from PIL import Image
import numpy as np


def build_expectations(df: pd.DataFrame) -> PandasDataset:
    ge_df = PandasDataset(df)
    ge_df.expect_column_values_to_not_be_null("path")
    ge_df.expect_column_values_to_be_unique("sha256")
    ge_df.expect_column_values_to_be_in_set("split", ["train", "val", "test", "unsplit"])
    ge_df.expect_column_values_to_be_in_set("class", df["class"].unique())
    ge_df.expect_column_values_to_match_regex("path", r".+\.(jpg|jpeg|png|bmp|gif)$")
    ge_df.expect_column_values_to_be_between("bytes", min_value=1)
    return ge_df


def brightness_sample(df: pd.DataFrame, n: int = 200) -> float:
    sample = df.sample(min(len(df), n), random_state=42)
    vals = []
    for p in sample["path"]:
        with Image.open(p) as img:
            gray = img.convert("L")
            vals.append(float(np.mean(gray)))
    return float(np.mean(vals)) if vals else None


def render_report(results: Dict, out_path: Path):
    renderer = ValidationResultsPageRenderer()
    document = renderer.render(results)
    html = DefaultJinjaPageView().render(document)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def load_metadata(processed_root: Path) -> Dict:
    meta_path = processed_root / "metadata.json"
    return json.loads(meta_path.read_text())


def compute_reference_stats(processed_manifest: pd.DataFrame) -> Dict:
    class_counts = processed_manifest["class"].value_counts(normalize=True).to_dict()
    brightness_mean = processed_manifest.get("brightness", pd.Series(dtype=float)).mean() if "brightness" in processed_manifest else None
    return {"class_distribution": class_counts, "brightness_mean": brightness_mean}


def compute_drift(current_stats: Dict, reference_stats: Dict, thresholds: Dict) -> List[str]:
    alerts = []
    ref_dist = reference_stats.get("class_distribution", {})
    cur_dist = current_stats.get("class_distribution", {})
    for cls, ref_p in ref_dist.items():
        cur_p = cur_dist.get(cls, 0.0)
        if ref_p == 0:
            continue
        if abs(cur_p - ref_p) > thresholds.get("class_pct", 0.2):
            alerts.append(f"Class distribution drift for {cls}: {ref_p:.3f} -> {cur_p:.3f}")

    ref_b = reference_stats.get("brightness_mean")
    cur_b = current_stats.get("brightness_mean")
    if ref_b is not None and cur_b is not None:
        if abs(cur_b - ref_b) > thresholds.get("brightness", 10):
            alerts.append(f"Brightness mean drift: {ref_b:.2f} -> {cur_b:.2f}")
    return alerts


def notify(alerts: List[str]):
    webhook = os.getenv("ALERT_WEBHOOK")
    if webhook and alerts:
        try:
            requests.post(webhook, json={"text": "\n".join(alerts)}, timeout=10)
        except Exception:  # noqa: BLE001
            pass


def main():
    parser = argparse.ArgumentParser(description="Run Great Expectations profiling and drift checks")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports_dir", type=Path, default=Path("reports/ge"))
    parser.add_argument("--reference", type=Path, help="Optional reference stats JSON for drift detection")
    parser.add_argument("--class_pct_threshold", type=float, default=0.2)
    parser.add_argument("--brightness_threshold", type=float, default=10.0)
    args = parser.parse_args()

    processed_root = args.processed_dir / args.dataset
    manifest_path = processed_root / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Processed manifest not found: {manifest_path}. Run prepare first.")

    df = pd.read_csv(manifest_path)
    ge_df = build_expectations(df)
    results = ge_df.validate()

    report_path = args.reports_dir / args.dataset / "index.html"
    render_report(results, report_path)

    stats = {
        "class_distribution": df["class"].value_counts(normalize=True).to_dict(),
        "brightness_mean": brightness_sample(df),
    }

    reference_stats = {}
    if args.reference and args.reference.exists():
        reference_stats = json.loads(args.reference.read_text())
    thresholds = {"class_pct": args.class_pct_threshold, "brightness": args.brightness_threshold}
    alerts = compute_drift(stats, reference_stats, thresholds) if reference_stats else []
    notify(alerts)

    drift_report = {
        "alerts": alerts,
        "current": stats,
        "reference": reference_stats,
        "thresholds": thresholds,
    }
    drift_path = args.reports_dir / args.dataset / "drift.json"
    drift_path.write_text(json.dumps(drift_report, indent=2))

    print(f"Validation report saved to {report_path}")
    if alerts:
        print("DRIFT ALERTS:")
        for a in alerts:
            print(f"- {a}")
    else:
        print("No drift detected")


if __name__ == "__main__":
    main()
