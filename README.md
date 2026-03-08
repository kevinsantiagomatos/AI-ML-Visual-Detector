# AI Vision Demo

End-to-end computer vision starter: classifier + optional YOLO detector, FastAPI backend, minimal web UI, Docker deploy.

## Quickstart (local CPU)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

## Train classifier (transfer learning)
Prepare an ImageFolder dataset:
```
data/
  train/
    cat/...
    dog/...
    fox/...
  val/
    cat/...
    dog/...
    fox/...
```
Then run:
```bash
python -m src.train_classifier --data data --epochs 5 --output models/pet_classifier.pt
```
- Uses ResNet50 pretrained on ImageNet, only fine-tunes the last layer.
- Saves best checkpoint + `metrics.json` in `models/`.

## Optional: train detector (YOLOv8)
Label boxes with CVAT/Label Studio and export YOLO/COCO YAML. Then:
```bash
python -m src.train_detector --data data/pets.yaml --base yolov8n.pt --epochs 50 --output models/yolo-detector.pt
```
If you skip this step, the app will still run `/detect` using the default COCO-pretrained `yolov8n.pt` downloaded on startup.

## API
- `POST /predict` → `[{"label": str, "score": float}]`
- `POST /detect` → `{ "objects": [{"bbox": [x1,y1,x2,y2], "label": str, "score": float}] }`
- `GET /health`
- `GET /config` → current model + runtime limits

Environment vars:
- `CLASSIFIER_WEIGHTS` (default `models/pet_classifier.pt`)
- `DETECTOR_WEIGHTS` (default `models/yolo-detector.pt`)
- `DETECTOR_FALLBACK` (default `yolov8n.pt`, downloaded automatically)
- `MAX_IMAGE_MB` (default `8`)
- `TOPK` (default `3`)

## Docker
```bash
docker build -t vision-demo .
docker run -p 8000:8000 -e CLASSIFIER_WEIGHTS=/app/models/pet_classifier.pt vision-demo
```
Mount custom weights:
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models vision-demo
```

## Deployment (AWS idea)
- Push image to ECR.
- Run on ECS Fargate with ALB; set env vars for weight paths.
- Store weights in S3 and download on task start if large.

## Notes
- If no custom classifier weights are present, the API falls back to ImageNet ResNet50 and returns top-3 ImageNet labels.
- Detection endpoint returns 503 unless YOLO weights are provided or the fallback model loads.
