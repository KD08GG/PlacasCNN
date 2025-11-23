# train_yolo.py
from ultralytics import YOLO
from config import DATA_YAML, YOLO_TRAIN_CONFIG, create_directories, PRETRAINED_YOLO
from pathlib import Path

def train_yolo(data_yaml=None, epochs=None, imgsz=None, batch=None, device=None, project=None, name=None):
    create_directories()
    data_yaml = data_yaml or str(DATA_YAML)
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"No se encontró {data_yaml}")
    epochs = epochs or YOLO_TRAIN_CONFIG["epochs"]
    imgsz = imgsz or YOLO_TRAIN_CONFIG["imgsz"]
    batch = batch or YOLO_TRAIN_CONFIG["batch"]
    project = project or YOLO_TRAIN_CONFIG["project"]
    name = name or YOLO_TRAIN_CONFIG["name"]
    device = device or YOLO_TRAIN_CONFIG["device"]
    print("Iniciando entrenamiento YOLO...")
    model = YOLO(PRETRAINED_YOLO)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=project, name=name)
    print("Entrenamiento YOLO finalizado.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="ruta a data.yaml")
    args = parser.parse_args()
    train_yolo(data_yaml=args.data)
