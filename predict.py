# predict.py
import argparse
from pipeline.alpr_pipeline import ALPRPipeline
from config import create_directories, CLASSIFIER_MODEL_DIR
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="ruta a imagen")
    parser.add_argument("--classifier", type=str, help="ruta a modelo classifier (.h5). Si no, busca models/classifier/classifier.h5")
    parser.add_argument("--no-easyocr", action="store_true", help="desactivar fallback easyocr")
    args = parser.parse_args()
    create_directories()
    classifier_path = args.classifier or str(Path(CLASSIFIER_MODEL_DIR) / "classifier.h5")
    pipeline = ALPRPipeline(classifier_model=classifier_path, use_easyocr=not args.no_easyocr)
    results = pipeline.recognize_from_path(args.image, save_crops=True, visualize=True)
    print("RESULTADOS:", results)

if __name__ == "__main__":
    main()
