"""Fine-tune a YOLOv8 detector with ultralytics.
Expect a YOLO dataset YAML (train/val image/label paths). Example:
  python -m src.train_detector --data data/pets.yaml --epochs 50 --output models/yolo-pets.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def train(args):
    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="finetune",
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        out_path.write_bytes(best.read_bytes())
        print(f"Saved fine-tuned weights to {out_path}")
    else:
        print("Could not find best.pt; check ultralytics output")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--base", default="yolov8n.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default="models/yolo-detector.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
