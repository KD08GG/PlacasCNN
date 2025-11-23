# detectors/plate_detector.py
from ultralytics import YOLO
import cv2
from pathlib import Path
from config import YOLO_MODEL_DIR, PRETRAINED_YOLO, YOLO_TRAIN_CONFIG

class PlateDetector:
    def __init__(self, model_path=None):
        # si no pasa ruta, intenta usar best.pt de entrenamiento
        if model_path is None:
            candidate = Path(YOLO_TRAIN_CONFIG["project"]) / YOLO_TRAIN_CONFIG["name"] / "weights" / "best.pt"
            if candidate.exists():
                model_path = str(candidate)
            else:
                model_path = PRETRAINED_YOLO
        print(f"[PlateDetector] cargando modelo: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, image, conf=None, iou=None):
        conf = conf or YOLO_TRAIN_CONFIG.get("conf", None)
        iou = iou or YOLO_TRAIN_CONFIG.get("iou", None)
        # model returns Results object list; pass image (numpy) or path
        results = self.model(image, conf=conf or 0.45, iou=iou or 0.45)
        boxes = results[0].boxes
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2].copy()
            crops.append({
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "conf": float(box.conf[0])
            })
        return crops
