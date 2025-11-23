# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

# Dataset / models / outputs
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"           # carpeta donde se extrae el zip Roboflow
DATA_YAML = DATASET_DIR / "data.yaml"        # Roboflow data.yaml (si aplica)

MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_DIR = MODELS_DIR / "yolo"
CLASSIFIER_MODEL_DIR = MODELS_DIR / "classifier"
SEGMENTER_MODEL_DIR = MODELS_DIR / "segmenter"

RESULTS_DIR = PROJECT_ROOT / "results"
CROPS_DIR = RESULTS_DIR / "crops"

# YOLO / training defaults
PRETRAINED_YOLO = "yolov8n.pt"
YOLO_TRAIN_CONFIG = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 8,
    "project": str(YOLO_MODEL_DIR),
    "name": "exp1",
    "device": None,  # si None el script detecta GPU
}

# Classifier
CLASSIFIER_CONFIG = {
    "img_size": 32,
    "batch": 64,
    "epochs": 50,
    "num_classes": 36,  # 0-9 + A-Z
    # mapping index -> char
    "class_map": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

# Segmenter (if usar CNN)
SEGMENTER_CONFIG = {
    "img_size": 256,
    "batch": 8,
    "epochs": 50,
}

# OCR pipeline
ALPR_CONFIG = {
    "yolo_conf": 0.4,
    "yolo_iou": 0.45,
    "save_crops": True,
    "use_easyocr_fallback": True,
}

def create_directories():
    for p in [
        DATA_DIR, DATASET_DIR, MODELS_DIR, YOLO_MODEL_DIR,
        CLASSIFIER_MODEL_DIR, SEGMENTER_MODEL_DIR, RESULTS_DIR, CROPS_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)
