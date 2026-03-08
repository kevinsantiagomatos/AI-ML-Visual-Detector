import os
from pathlib import Path
from typing import List

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.inference import Classifier, Prediction, detect_objects, load_detector

app = FastAPI(title="CV Classifier/Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS", "models/pet_classifier.pt")
DETECTOR_WEIGHTS = os.getenv("DETECTOR_WEIGHTS", "models/yolo-detector.pt")
DETECTOR_FALLBACK = os.getenv("DETECTOR_FALLBACK", "yolov8n.pt")
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "8"))
TOPK = int(os.getenv("TOPK", "3"))


@app.on_event("startup")
def _load_models():
    global classifier
    classifier = Classifier(
        model_path=CLASSIFIER_WEIGHTS if Path(CLASSIFIER_WEIGHTS).exists() else None,
        topk=TOPK,
    )
    global detector
    detector = None
    try:
        if Path(DETECTOR_WEIGHTS).exists():
            detector = load_detector(DETECTOR_WEIGHTS)
        else:
            detector = load_detector(DETECTOR_FALLBACK)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load detector: {e}")


@app.get("/", response_class=HTMLResponse)
def index():
    if not STATIC_DIR.exists():
        return HTMLResponse("<h1>Upload /predict via POST</h1>")
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> List[Prediction]:
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        preds = await anyio.to_thread.run_sync(classifier.predict, image_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return [p.__dict__ for p in preds]


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector weights not loaded")
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        objects = await anyio.to_thread.run_sync(detect_objects, detector, image_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return {"objects": objects}


@app.get("/config")
def config():
    return {
        "classifier_weights": CLASSIFIER_WEIGHTS if Path(CLASSIFIER_WEIGHTS).exists() else "imagenet_resnet50",
        "detector_weights": DETECTOR_WEIGHTS if Path(DETECTOR_WEIGHTS).exists() else DETECTOR_FALLBACK,
        "max_image_mb": MAX_IMAGE_MB,
        "topk": TOPK,
    }
