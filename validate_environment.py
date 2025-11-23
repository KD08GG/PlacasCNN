# validate_environment.py
"""
Script para validar que el entorno esté configurado correctamente.
Verifica dependencias, modelos, y configuración.
"""
import sys
from pathlib import Path
import importlib

def check_python_version():
    """Verifica versión de Python."""
    version = sys.version_info
    print("\n" + "="*70)
    print("VERIFICACIÓN DE ENTORNO - Sistema ALPR")
    print("="*70)
    print(f"\n1. Python")
    print("-" * 70)

    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (Se requiere 3.8+)")
        return False

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print(f"\n2. Dependencias")
    print("-" * 70)

    dependencies = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "tensorflow": "tensorflow",
        "ultralytics": "ultralytics",
        "easyocr": "easyocr",
        "PIL": "Pillow",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
        "tqdm": "tqdm"
    }

    all_ok = True
    for module_name, package_name in dependencies.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "?")
            print(f"   ✓ {package_name:20s} ({version})")
        except ImportError:
            print(f"   ✗ {package_name:20s} NO INSTALADO")
            all_ok = False

    return all_ok

def check_gpu():
    """Verifica disponibilidad de GPU."""
    print(f"\n3. GPU")
    print("-" * 70)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"   ✓ GPU disponible: {gpu_name}")
            print(f"   ✓ CUDA Version: {cuda_version}")
            return True
        else:
            print("   ⚠ GPU no disponible - se usará CPU")
            print("   (El entrenamiento será más lento)")
            return False
    except ImportError:
        print("   ⚠ PyTorch no instalado - no se puede verificar GPU")
        return False

def check_project_structure():
    """Verifica estructura de directorios del proyecto."""
    print(f"\n4. Estructura del Proyecto")
    print("-" * 70)

    required_dirs = [
        "detectors",
        "segmenters",
        "recognizers",
        "pipeline",
        "utils"
    ]

    required_files = [
        "config.py",
        "setup_dataset.py",
        "train_yolo.py",
        "train_classifier.py",
        "predict.py",
        "evaluate.py",
        "extract_chars.py"
    ]

    all_ok = True

    # Verificar directorios
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"   ✓ {dir_name}/")
        else:
            print(f"   ✗ {dir_name}/ NO ENCONTRADO")
            all_ok = False

    # Verificar archivos
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"   ✓ {file_name}")
        else:
            print(f"   ✗ {file_name} NO ENCONTRADO")
            all_ok = False

    return all_ok

def check_config():
    """Verifica que la configuración sea válida."""
    print(f"\n5. Configuración")
    print("-" * 70)

    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
            YOLO_TRAIN_CONFIG, CLASSIFIER_CONFIG, ALPR_CONFIG,
            create_directories
        )

        print(f"   ✓ config.py importado correctamente")
        print(f"   ✓ PROJECT_ROOT: {PROJECT_ROOT}")

        # Crear directorios si no existen
        create_directories()
        print(f"   ✓ Directorios creados/verificados")

        return True
    except Exception as e:
        print(f"   ✗ Error en configuración: {e}")
        return False

def check_models():
    """Verifica disponibilidad de modelos."""
    print(f"\n6. Modelos")
    print("-" * 70)

    from config import YOLO_MODEL_DIR, CLASSIFIER_MODEL_DIR, YOLO_TRAIN_CONFIG

    # Verificar YOLO
    yolo_path = Path(YOLO_TRAIN_CONFIG["project"]) / YOLO_TRAIN_CONFIG["name"] / "weights" / "best.pt"
    if yolo_path.exists():
        print(f"   ✓ Modelo YOLO entrenado: {yolo_path}")
    else:
        print(f"   ⚠ Modelo YOLO no entrenado: {yolo_path}")
        print(f"     Ejecuta: python train_yolo.py --data data/dataset/data.yaml")

    # Verificar clasificador
    classifier_path = Path(CLASSIFIER_MODEL_DIR) / "classifier.h5"
    if classifier_path.exists():
        print(f"   ✓ Modelo clasificador: {classifier_path}")
    else:
        print(f"   ⚠ Modelo clasificador no entrenado: {classifier_path}")
        print(f"     Ejecuta: python train_classifier.py --data data/char_dataset")

    return True

def check_dataset():
    """Verifica disponibilidad de datasets."""
    print(f"\n7. Datasets")
    print("-" * 70)

    from config import DATASET_DIR, DATA_DIR

    # Dataset de detección
    dataset_path = Path(DATASET_DIR)
    if dataset_path.exists():
        train_path = dataset_path / "train" / "images"
        if train_path.exists() and any(train_path.iterdir()):
            count = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
            print(f"   ✓ Dataset de detección: {count} imágenes de entrenamiento")
        else:
            print(f"   ⚠ Dataset de detección vacío o incompleto")
    else:
        print(f"   ⚠ Dataset de detección no encontrado: {dataset_path}")
        print(f"     Ejecuta: python -c \"from setup_dataset import unzip_dataset; unzip_dataset('path/to/dataset.zip')\"")

    # Dataset de caracteres
    char_dataset = Path(DATA_DIR) / "char_dataset"
    if char_dataset.exists():
        classes = [d for d in char_dataset.iterdir() if d.is_dir()]
        if classes:
            total_images = sum(len(list(c.glob("*.png"))) + len(list(c.glob("*.jpg"))) for c in classes)
            print(f"   ✓ Dataset de caracteres: {len(classes)} clases, {total_images} imágenes")
        else:
            print(f"   ⚠ Dataset de caracteres vacío")
    else:
        print(f"   ⚠ Dataset de caracteres no encontrado: {char_dataset}")
        print(f"     Ejecuta: python extract_chars.py --images carpeta_imagenes")
        print(f"     O genera sintético: python generate_synthetic_chars.py")

    return True

def run_tests():
    """Ejecuta tests unitarios."""
    print(f"\n8. Tests Unitarios")
    print("-" * 70)

    try:
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')

        if suite.countTestCases() == 0:
            print("   ⚠ No se encontraron tests")
            return True

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        if result.wasSuccessful():
            print(f"   ✓ Todos los tests pasaron ({result.testsRun} tests)")
            return True
        else:
            print(f"   ✗ Algunos tests fallaron")
            print(f"     Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
            print(f"     Fallidos: {len(result.failures)}")
            print(f"     Errores: {len(result.errors)}")
            return False
    except Exception as e:
        print(f"   ✗ Error ejecutando tests: {e}")
        return False

def print_summary(results):
    """Imprime resumen de la validación."""
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)

    checks = [
        ("Python", results["python"]),
        ("Dependencias", results["dependencies"]),
        ("GPU", results["gpu"]),
        ("Estructura", results["structure"]),
        ("Configuración", results["config"]),
        ("Modelos", results["models"]),
        ("Datasets", results["datasets"]),
        ("Tests", results["tests"])
    ]

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    for name, status in checks:
        icon = "✓" if status else "✗"
        print(f"   {icon} {name}")

    print(f"\nResultado: {passed}/{total} verificaciones pasadas")

    if passed == total:
        print("\n✅ Entorno completamente configurado!")
        print("\nPuedes ejecutar:")
        print("  - python train_yolo.py --data data/dataset/data.yaml")
        print("  - python train_classifier.py --data data/char_dataset")
        print("  - python predict.py --image examples/test.jpg")
    else:
        print("\n⚠️  Algunas verificaciones fallaron")
        print("Revisa los mensajes anteriores para corregir problemas")

    print("="*70 + "\n")

def main():
    """Función principal."""
    results = {}

    results["python"] = check_python_version()
    results["dependencies"] = check_dependencies()
    results["gpu"] = check_gpu()
    results["structure"] = check_project_structure()
    results["config"] = check_config()
    results["models"] = check_models()
    results["datasets"] = check_dataset()
    results["tests"] = run_tests()

    print_summary(results)

if __name__ == "__main__":
    main()
