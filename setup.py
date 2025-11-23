# setup.py
"""
Script de instalación y configuración del proyecto ALPR.
"""
import subprocess
import sys
from pathlib import Path
from config import create_directories

def check_python_version():
    """Verifica versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Error: Se requiere Python 3.8+, tienes {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Instala dependencias desde requirements.txt."""
    print("\nInstalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_directories():
    """Crea estructura de directorios."""
    print("\nCreando directorios...")
    try:
        create_directories()
        print("✓ Directorios creados correctamente")
        return True
    except Exception as e:
        print(f"❌ Error creando directorios: {e}")
        return False

def check_gpu():
    """Verifica si hay GPU disponible."""
    print("\nVerificando GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No se detectó GPU, se usará CPU")
            return False
    except ImportError:
        print("⚠ PyTorch no instalado, no se puede verificar GPU")
        return False

def download_pretrained_yolo():
    """Descarga modelo preentrenado de YOLOv8."""
    print("\nDescargando modelo YOLOv8 preentrenado...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("✓ Modelo YOLOv8n descargado")
        return True
    except Exception as e:
        print(f"❌ Error descargando YOLOv8: {e}")
        return False

def verify_installation():
    """Verifica que los módulos principales se importen correctamente."""
    print("\nVerificando instalación...")
    modules = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"),
        ("ultralytics", "Ultralytics"),
        ("easyocr", "EasyOCR"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("tqdm", "tqdm")
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} no disponible")
            all_ok = False

    return all_ok

def create_example_files():
    """Crea archivos de ejemplo."""
    print("\nCreando archivos de ejemplo...")

    # Crear ground_truth.json de ejemplo
    example_gt = {
        "image1.jpg": "ABC123D",
        "image2.jpg": "XYZ789E"
    }

    import json
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    gt_file = examples_dir / "ground_truth_example.json"
    with open(gt_file, 'w') as f:
        json.dump(example_gt, f, indent=2)

    print(f"  ✓ {gt_file}")

    # Crear README en examples
    readme_content = """# Ejemplos

Este directorio contiene ejemplos para probar el sistema ALPR.

## Uso

1. Coloca tus imágenes de prueba aquí
2. Ejecuta: `python predict.py --image examples/tu_imagen.jpg`
3. Los resultados se guardarán en `results/crops/`

## Ground Truth

Para evaluación, crea un archivo `ground_truth.json` con formato:
```json
{
  "imagen1.jpg": "ABC123D",
  "imagen2.jpg": "XYZ789E"
}
```

Luego ejecuta:
```bash
python evaluate.py --images examples/ --ground-truth examples/ground_truth.json
```
"""

    readme_file = examples_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"  ✓ {readme_file}")

    return True

def print_next_steps():
    """Imprime próximos pasos."""
    print("\n" + "="*70)
    print("INSTALACIÓN COMPLETADA")
    print("="*70)
    print("\nPróximos pasos:\n")
    print("1. Preparar dataset de detección:")
    print("   python -c \"from setup_dataset import unzip_dataset; unzip_dataset('ruta/al/dataset.zip')\"")
    print("\n2. Entrenar detector YOLOv8:")
    print("   python train_yolo.py --data data/dataset/data.yaml")
    print("\n3. Preparar dataset de caracteres:")
    print("   - Estructura: data/char_dataset/0/, 1/, ..., A/, B/, ...")
    print("   - O usar: python extract_chars.py --images carpeta_imagenes --output data/char_dataset")
    print("\n4. Entrenar clasificador:")
    print("   python train_classifier.py --data data/char_dataset --epochs 50")
    print("\n5. Hacer predicciones:")
    print("   python predict.py --image ruta/imagen.jpg")
    print("\n6. Evaluar sistema:")
    print("   python evaluate.py --images test_images/ --ground-truth ground_truth.json")
    print("\nPara más información, consulta el README.md")
    print("="*70 + "\n")

def main():
    """Función principal de setup."""
    print("="*70)
    print("SETUP - Sistema ALPR")
    print("="*70)

    # Verificar Python
    if not check_python_version():
        return

    # Instalar dependencias
    if not install_requirements():
        print("\n⚠ Algunas dependencias no se instalaron correctamente")
        print("Puedes intentar instalarlas manualmente:")
        print("  pip install -r requirements.txt")

    # Crear directorios
    setup_directories()

    # Verificar GPU
    check_gpu()

    # Descargar YOLOv8
    download_pretrained_yolo()

    # Verificar instalación
    if not verify_installation():
        print("\n⚠ Algunos módulos no están disponibles")
        print("Revisa los errores anteriores e instala los paquetes faltantes")

    # Crear ejemplos
    create_example_files()

    # Imprimir próximos pasos
    print_next_steps()

if __name__ == "__main__":
    main()
