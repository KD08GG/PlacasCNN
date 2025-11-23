# train.py - Entrenamiento simplificado
"""
Script simple para entrenar detector y clasificador.
"""
import argparse
from pathlib import Path

# ============================================================================
# ENTRENAR YOLO
# ============================================================================

def train_detector(data_yaml, epochs=50, img_size=640):
    """Entrena detector YOLOv8."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("ENTRENANDO DETECTOR YOLO")
    print(f"{'='*60}\n")

    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=8,
        project="models/yolo",
        name="detector"
    )

    print("\n✓ Detector entrenado!")
    print("  Modelo guardado en: models/yolo/detector/weights/best.pt\n")

# ============================================================================
# ENTRENAR CLASIFICADOR
# ============================================================================

def build_classifier_model(num_classes=36, img_size=32):
    """Crea modelo CNN simple."""
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input((img_size, img_size, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_classifier(data_dir, epochs=50, img_size=32):
    """Entrena clasificador CNN."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print(f"\n{'='*60}")
    print("ENTRENANDO CLASIFICADOR CNN")
    print(f"{'='*60}\n")

    # Data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        rotation_range=8,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    # Generadores
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        subset='validation'
    )

    # Crear y entrenar modelo
    model = build_classifier_model(num_classes=36, img_size=img_size)
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    # Guardar
    output_path = Path("models/classifier.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    print(f"\n✓ Clasificador entrenado!")
    print(f"  Modelo guardado en: {output_path}\n")

# ============================================================================
# GENERAR DATOS SINTÉTICOS
# ============================================================================

def generate_synthetic_data(output_dir, samples=100):
    """Genera dataset sintético de caracteres."""
    import cv2
    import numpy as np
    from tqdm import tqdm

    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print("GENERANDO DATASET SINTÉTICO")
    print(f"{'='*60}\n")
    print(f"Clases: {len(chars)}")
    print(f"Muestras por clase: {samples}")
    print(f"Total: {len(chars) * samples}\n")

    for char in tqdm(chars, desc="Generando"):
        class_dir = output_dir / char
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples):
            # Crear imagen
            img = np.ones((32, 32), dtype=np.uint8) * 255

            # Dibujar carácter
            cv2.putText(img, char, (4, 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

            # Augmentación simple
            if np.random.rand() > 0.5:
                # Ruido
                noise = np.random.normal(0, 25, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)

            if np.random.rand() > 0.5:
                # Rotación pequeña
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((16, 16), angle, 1)
                img = cv2.warpAffine(img, M, (32, 32), borderValue=255)

            # Guardar
            cv2.imwrite(str(class_dir / f"{char}_{i:04d}.png"), img)

    print(f"\n✓ Dataset generado en: {output_dir}\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento simplificado")

    subparsers = parser.add_subparsers(dest='command', help='Comando')

    # Detector
    parser_det = subparsers.add_parser('detector', help='Entrenar detector')
    parser_det.add_argument('--data', required=True, help='Ruta a data.yaml')
    parser_det.add_argument('--epochs', type=int, default=50)

    # Clasificador
    parser_clf = subparsers.add_parser('classifier', help='Entrenar clasificador')
    parser_clf.add_argument('--data', required=True, help='Directorio con clases')
    parser_clf.add_argument('--epochs', type=int, default=50)

    # Sintético
    parser_syn = subparsers.add_parser('synthetic', help='Generar datos sintéticos')
    parser_syn.add_argument('--output', default='data/synthetic_chars')
    parser_syn.add_argument('--samples', type=int, default=100)

    args = parser.parse_args()

    if args.command == 'detector':
        train_detector(args.data, epochs=args.epochs)

    elif args.command == 'classifier':
        train_classifier(args.data, epochs=args.epochs)

    elif args.command == 'synthetic':
        generate_synthetic_data(args.output, samples=args.samples)

    else:
        parser.print_help()
        print("\nEjemplos:")
        print("  python train.py synthetic --samples 100")
        print("  python train.py detector --data data/dataset/data.yaml")
        print("  python train.py classifier --data data/chars")

if __name__ == "__main__":
    main()
