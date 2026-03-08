"""Transfer-learn a ResNet50 classifier on an ImageFolder dataset.
Usage:
  python -m src.train_classifier --data data/pets --epochs 5 --output models/pet_classifier.pt
The script expects an ImageFolder with train/val splits or a single folder; if only one
folder is provided, it will create a random 80/20 split on the fly.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def build_dataloaders(data_dir: Path, batch_size: int = 32, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, List[str]]:
    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    has_split = (data_dir / "train").exists() and (data_dir / "val").exists()
    if has_split:
        train_ds = datasets.ImageFolder(data_dir / "train", transform=transform_train)
        val_ds = datasets.ImageFolder(data_dir / "val", transform=transform_val)
    else:
        # Build two copies so train/val can use different transforms
        full_ds = datasets.ImageFolder(data_dir)
        indices = torch.randperm(len(full_ds))
        val_size = max(1, int(0.2 * len(full_ds)))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        train_ds = torch.utils.data.Subset(
            datasets.ImageFolder(data_dir, transform=transform_train), train_idx.tolist()
        )
        val_ds = torch.utils.data.Subset(
            datasets.ImageFolder(data_dir, transform=transform_val), val_idx.tolist()
        )

    class_names = getattr(train_ds, "classes", None) or getattr(train_ds.dataset, "classes")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_dataloaders(Path(args.data), args.batch_size, args.num_workers)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_state = {
        "state_dict": model.state_dict(),
        "metadata": {"class_names": class_names, "arch": "resnet50", "val_acc": 0.0},
    }

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_acc / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_acc += accuracy(outputs, labels) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {
                "state_dict": model.state_dict(),
                "metadata": {
                    "class_names": class_names,
                    "arch": "resnet50",
                    "val_acc": val_acc,
                },
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)

    metrics = {"best_val_acc": best_acc, "num_classes": len(class_names)}
    (output_path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved best model to {output_path} with val_acc={best_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 for image classification")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root (ImageFolder)")
    parser.add_argument("--output", type=str, default="models/pet_classifier.pt", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
