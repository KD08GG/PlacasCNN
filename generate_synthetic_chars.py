# generate_synthetic_chars.py
"""
Genera dataset sintético de caracteres para entrenamiento rápido.
Útil para pruebas y desarrollo.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from config import CLASSIFIER_CONFIG, create_directories

def generate_char_image(char, img_size=32, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Genera imagen sintética de un carácter.

    Args:
        char: Carácter a generar
        img_size: Tamaño de la imagen
        font: Fuente de OpenCV

    Returns:
        Imagen numpy array
    """
    # Crear imagen en blanco
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255

    # Calcular tamaño y posición del texto
    font_scale = 0.8
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(char, font, font_scale, thickness)

    # Centrar texto
    x = (img_size - text_width) // 2
    y = (img_size + text_height) // 2

    # Dibujar texto
    cv2.putText(img, char, (x, y), font, font_scale, 0, thickness)

    return img

def add_noise(img, noise_level=0.1):
    """Agrega ruido gaussiano a la imagen."""
    noise = np.random.normal(0, noise_level * 255, img.shape)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img

def add_rotation(img, angle_range=15):
    """Rota la imagen aleatoriamente."""
    angle = np.random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return rotated

def add_perspective(img, strength=0.1):
    """Aplica transformación de perspectiva."""
    h, w = img.shape[:2]

    # Puntos originales
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Puntos transformados con variación aleatoria
    offset = int(w * strength)
    pts2 = np.float32([
        [np.random.randint(0, offset), np.random.randint(0, offset)],
        [w - np.random.randint(0, offset), np.random.randint(0, offset)],
        [np.random.randint(0, offset), h - np.random.randint(0, offset)],
        [w - np.random.randint(0, offset), h - np.random.randint(0, offset)]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w, h), borderValue=255)
    return warped

def augment_image(img):
    """Aplica augmentación aleatoria."""
    # Probabilidad de aplicar cada transformación
    if np.random.rand() > 0.5:
        img = add_noise(img, noise_level=np.random.uniform(0.05, 0.15))

    if np.random.rand() > 0.5:
        img = add_rotation(img, angle_range=np.random.uniform(5, 15))

    if np.random.rand() > 0.3:
        img = add_perspective(img, strength=np.random.uniform(0.05, 0.15))

    # Blur
    if np.random.rand() > 0.5:
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return img

def generate_dataset(output_dir, samples_per_class=100, img_size=32, augment=True):
    """
    Genera dataset sintético completo.

    Args:
        output_dir: Directorio de salida
        samples_per_class: Número de muestras por clase
        img_size: Tamaño de imagen
        augment: Si aplicar augmentación
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chars = CLASSIFIER_CONFIG["class_map"]
    total = len(chars) * samples_per_class

    print(f"Generando dataset sintético...")
    print(f"  Clases: {len(chars)}")
    print(f"  Muestras por clase: {samples_per_class}")
    print(f"  Total: {total}")
    print(f"  Augmentación: {'Sí' if augment else 'No'}\n")

    with tqdm(total=total, desc="Generando") as pbar:
        for char in chars:
            # Crear directorio de clase
            class_dir = output_dir / char
            class_dir.mkdir(exist_ok=True)

            for i in range(samples_per_class):
                # Generar imagen base
                img = generate_char_image(char, img_size=img_size)

                # Aplicar augmentación
                if augment:
                    img = augment_image(img)

                # Guardar
                filename = class_dir / f"{char}_{i:04d}.png"
                cv2.imwrite(str(filename), img)

                pbar.update(1)

    print(f"\n✓ Dataset generado en: {output_dir}")
    print(f"  Verifica el dataset antes de entrenar")

def verify_dataset(dataset_dir):
    """Verifica que el dataset esté completo."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} no existe")
        return False

    chars = CLASSIFIER_CONFIG["class_map"]
    print(f"\nVerificando dataset...")

    all_ok = True
    for char in chars:
        class_dir = dataset_dir / char
        if not class_dir.exists():
            print(f"  ✗ Falta clase: {char}")
            all_ok = False
            continue

        count = len(list(class_dir.glob("*.png")))
        if count == 0:
            print(f"  ✗ Clase {char} vacía")
            all_ok = False
        else:
            print(f"  ✓ Clase {char}: {count} imágenes")

    if all_ok:
        print("\n✓ Dataset verificado correctamente")
    else:
        print("\n✗ Dataset tiene problemas")

    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Generar dataset sintético de caracteres")
    parser.add_argument("--output", default="data/char_dataset_synthetic", help="Directorio de salida")
    parser.add_argument("--samples", type=int, default=100, help="Muestras por clase")
    parser.add_argument("--size", type=int, default=32, help="Tamaño de imagen")
    parser.add_argument("--no-augment", action="store_true", help="Desactivar augmentación")
    parser.add_argument("--verify", action="store_true", help="Solo verificar dataset existente")

    args = parser.parse_args()

    create_directories()

    if args.verify:
        verify_dataset(args.output)
    else:
        generate_dataset(
            args.output,
            samples_per_class=args.samples,
            img_size=args.size,
            augment=not args.no_augment
        )
        verify_dataset(args.output)

        print("\n" + "="*70)
        print("Dataset sintético generado!")
        print("="*70)
        print("\nPróximos pasos:")
        print(f"1. Revisar visualmente algunas imágenes en {args.output}")
        print(f"2. Entrenar el clasificador:")
        print(f"   python train_classifier.py --data {args.output} --epochs 20")
        print("\nNota: Este dataset es sintético. Para mejores resultados,")
        print("entrena con datos reales extraídos de placas.")
        print("="*70)

if __name__ == "__main__":
    main()
