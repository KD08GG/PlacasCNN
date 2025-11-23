# setup_dataset.py
import zipfile
from pathlib import Path
from config import DATASET_DIR, create_directories

def unzip_dataset(zip_path, extract_to=None):
    extract_to = extract_to or DATASET_DIR
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError("Se requiere un .zip")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"Dataset extraído en {extract_to}")

def verify_dataset_structure(dataset_path=None):
    dataset_path = Path(dataset_path or DATASET_DIR)
    if not dataset_path.exists():
        print("Dataset no existe")
        return False
    # Chequeos mínimos (train/valid/images etc.)
    ok = True
    for s in ["train", "valid", "test"]:
        p = dataset_path / s / "images"
        if not p.exists() or not any(p.iterdir()):
            print(f"Advertencia: {p} vacío o no existe")
            ok = False
    print("Verificación completada")
    return ok

if __name__ == "__main__":
    create_directories()
    print("Uso: llamar unzip_dataset(path) o ejecutar como módulo con funciones importadas.")
