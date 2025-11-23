# extract_chars.py
"""
Script para extraer caracteres automáticamente de placas detectadas.
Útil para construir dataset de entrenamiento del clasificador.
"""
import cv2
import numpy as np
from pathlib import Path
from detectors.plate_detector import PlateDetector
from segmenters.classical_segmenter import ClassicalSegmenter
from config import create_directories, DATASET_DIR, DATA_DIR
import argparse
from tqdm import tqdm
import shutil

class CharacterExtractor:
    def __init__(self, yolo_model=None):
        self.detector = PlateDetector(yolo_model)
        self.segmenter = ClassicalSegmenter()

    def extract_from_image(self, image_path, output_dir, save_plates=True):
        """
        Extrae caracteres de una imagen y los guarda en output_dir.

        Args:
            image_path: Ruta a la imagen
            output_dir: Directorio donde guardar los caracteres
            save_plates: Si guardar las placas detectadas

        Returns:
            Lista de rutas de caracteres guardados
        """
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"No se pudo cargar: {image_path}")
            return []

        plates = self.detector.detect(img)
        saved_chars = []

        for plate_idx, plate_data in enumerate(plates):
            plate_img = plate_data["crop"]

            # Guardar placa si se solicita
            if save_plates:
                plate_dir = Path(output_dir) / "plates"
                plate_dir.mkdir(parents=True, exist_ok=True)
                plate_path = plate_dir / f"{Path(image_path).stem}_plate_{plate_idx}.jpg"
                cv2.imwrite(str(plate_path), plate_img)

            # Segmentar caracteres
            chars = self.segmenter.segment(plate_img)

            # Guardar cada carácter
            for char_idx, char_img in enumerate(chars):
                # Crear directorio 'unlabeled' para caracteres sin etiquetar
                char_dir = Path(output_dir) / "unlabeled"
                char_dir.mkdir(parents=True, exist_ok=True)

                char_path = char_dir / f"{Path(image_path).stem}_p{plate_idx}_c{char_idx}.png"
                cv2.imwrite(str(char_path), char_img)
                saved_chars.append(str(char_path))

        return saved_chars

    def extract_from_directory(self, images_dir, output_dir, extensions=[".jpg", ".jpeg", ".png"]):
        """
        Extrae caracteres de todas las imágenes en un directorio.

        Args:
            images_dir: Directorio con imágenes
            output_dir: Directorio de salida
            extensions: Extensiones de archivo a procesar
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Buscar todas las imágenes
        image_files = []
        for ext in extensions:
            image_files.extend(images_dir.glob(f"**/*{ext}"))

        print(f"Encontradas {len(image_files)} imágenes")

        total_chars = 0
        for img_path in tqdm(image_files, desc="Procesando imágenes"):
            chars = self.extract_from_image(img_path, output_dir)
            total_chars += len(chars)

        print(f"\n✓ Extracción completada!")
        print(f"  Total de caracteres extraídos: {total_chars}")
        print(f"  Guardados en: {output_dir}/unlabeled/")
        print(f"\nPróximos pasos:")
        print(f"1. Revisa los caracteres en {output_dir}/unlabeled/")
        print(f"2. Organiza manualmente los caracteres en carpetas por clase (0-9, A-Z)")
        print(f"3. Usa organize_chars.py para ayudar con la organización")

def organize_chars_interactive(chars_dir, output_dir):
    """
    Modo interactivo para etiquetar caracteres.

    Args:
        chars_dir: Directorio con caracteres sin etiquetar
        output_dir: Directorio de salida organizado por clases
    """
    chars_dir = Path(chars_dir)
    output_dir = Path(output_dir)

    char_files = list(chars_dir.glob("*.png"))
    if not char_files:
        print("No se encontraron caracteres para organizar")
        return

    print(f"\n{'='*60}")
    print(f"MODO INTERACTIVO - Organización de Caracteres")
    print(f"{'='*60}")
    print(f"\nTotal de caracteres: {len(char_files)}")
    print(f"\nInstrucciones:")
    print(f"  - Escribe el carácter que ves (0-9, A-Z)")
    print(f"  - Escribe 's' para saltar")
    print(f"  - Escribe 'q' para salir")
    print(f"{'='*60}\n")

    for idx, char_file in enumerate(char_files, 1):
        # Mostrar imagen
        img = cv2.imread(str(char_file))
        if img is None:
            continue

        # Redimensionar para mejor visualización
        h, w = img.shape[:2]
        scale = 200 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        display_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Carácter - Presiona ESC para cerrar", display_img)
        cv2.waitKey(100)

        print(f"\n[{idx}/{len(char_files)}] {char_file.name}")
        label = input("Etiqueta (0-9, A-Z, 's'=saltar, 'q'=salir): ").strip().upper()

        if label == 'Q':
            print("\nSaliendo...")
            break
        elif label == 'S':
            continue
        elif label and (label.isalnum() and len(label) == 1):
            # Crear carpeta de clase
            class_dir = output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)

            # Copiar archivo
            dest_path = class_dir / char_file.name
            shutil.copy(str(char_file), str(dest_path))
            print(f"  ✓ Guardado en {class_dir.name}/")
        else:
            print("  ✗ Etiqueta inválida, saltando...")

    cv2.destroyAllWindows()
    print(f"\n✓ Organización completada!")
    print(f"  Caracteres organizados en: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extraer caracteres de imágenes de placas")
    parser.add_argument("--images", required=True, help="Directorio con imágenes o ruta a imagen")
    parser.add_argument("--output", default="data/char_dataset_extracted", help="Directorio de salida")
    parser.add_argument("--organize", action="store_true", help="Modo interactivo para organizar")
    parser.add_argument("--yolo-model", help="Ruta a modelo YOLO personalizado")

    args = parser.parse_args()

    create_directories()

    if args.organize:
        # Modo organización
        chars_dir = Path(args.images)
        if not chars_dir.exists():
            print(f"Error: {chars_dir} no existe")
            return
        organize_chars_interactive(chars_dir, args.output)
    else:
        # Modo extracción
        extractor = CharacterExtractor(yolo_model=args.yolo_model)

        images_path = Path(args.images)
        if images_path.is_file():
            # Una sola imagen
            extractor.extract_from_image(images_path, args.output)
        elif images_path.is_dir():
            # Directorio de imágenes
            extractor.extract_from_directory(images_path, args.output)
        else:
            print(f"Error: {images_path} no es un archivo o directorio válido")

if __name__ == "__main__":
    main()
