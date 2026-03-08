import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision import models, transforms

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() // 2))


@dataclass
class Prediction:
    label: str
    score: float


def _default_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier:
    """Wrapper for image classification inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        topk: int = 3,
        use_half: bool = True,
    ):
        self.device = _default_device(device)
        self.topk = topk
        self.use_half = use_half and self.device.type == "cuda"
        self.model, self.class_names = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        if self.use_half:
            self.model.half()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path: Optional[str]):
        if model_path:
            state = torch.load(model_path, map_location="cpu")
            metadata = state.get("metadata", {})
            class_names = metadata.get("class_names")
            arch = metadata.get("arch", "resnet50")
            if arch != "resnet50":
                raise ValueError(f"Unsupported architecture saved: {arch}")
            num_classes = len(class_names) if class_names else None
            model = models.resnet50(weights=None)
            if num_classes:
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(state["state_dict"])
            if not class_names:
                raise ValueError("Class names missing from checkpoint metadata")
            return model, class_names
        # Fallback to pretrained ImageNet model
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        class_names = weights.meta["categories"]
        return model, class_names

    def predict(self, image_bytes: bytes) -> List[Prediction]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.use_half:
            tensor = tensor.half()
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_scores, top_idxs = probs.topk(self.topk)
        return [Prediction(self.class_names[idx], float(score)) for score, idx in zip(top_scores, top_idxs)]


def load_detector(weights_path: str, device: Optional[str] = None):
    """Load YOLO detector lazily to avoid heavy import when unused."""
    from ultralytics import YOLO  # imported here to keep import time light

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if device:
        dev = device
    model = YOLO(weights_path)
    model.to(dev)
    return model


def detect_objects(detector, image_bytes: bytes) -> List[Dict[str, Any]]:
    import cv2  # noqa: F401

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = detector.predict(source=image, verbose=False)
    objects: List[Dict[str, Any]] = []
    for r in results:
        boxes = r.boxes
        names = r.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            objects.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": names.get(cls_idx, str(cls_idx)),
                    "score": conf,
                }
            )
    return objects
