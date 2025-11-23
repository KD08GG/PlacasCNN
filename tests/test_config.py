# tests/test_config.py
"""Tests para configuración."""
import unittest
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    YOLO_TRAIN_CONFIG, CLASSIFIER_CONFIG, ALPR_CONFIG,
    create_directories
)

class TestConfig(unittest.TestCase):
    """Tests para configuración."""

    def test_project_root_exists(self):
        """Verifica que PROJECT_ROOT sea un Path válido."""
        self.assertIsInstance(PROJECT_ROOT, Path)
        self.assertTrue(PROJECT_ROOT.exists())

    def test_config_variables(self):
        """Verifica que las variables de configuración existan."""
        self.assertIsInstance(DATA_DIR, Path)
        self.assertIsInstance(MODELS_DIR, Path)
        self.assertIsInstance(RESULTS_DIR, Path)

    def test_yolo_config(self):
        """Verifica configuración de YOLO."""
        self.assertIn("epochs", YOLO_TRAIN_CONFIG)
        self.assertIn("imgsz", YOLO_TRAIN_CONFIG)
        self.assertIn("batch", YOLO_TRAIN_CONFIG)
        self.assertGreater(YOLO_TRAIN_CONFIG["epochs"], 0)
        self.assertGreater(YOLO_TRAIN_CONFIG["imgsz"], 0)

    def test_classifier_config(self):
        """Verifica configuración de clasificador."""
        self.assertIn("img_size", CLASSIFIER_CONFIG)
        self.assertIn("num_classes", CLASSIFIER_CONFIG)
        self.assertIn("class_map", CLASSIFIER_CONFIG)
        self.assertEqual(len(CLASSIFIER_CONFIG["class_map"]), CLASSIFIER_CONFIG["num_classes"])

    def test_alpr_config(self):
        """Verifica configuración de ALPR."""
        self.assertIn("yolo_conf", ALPR_CONFIG)
        self.assertIn("yolo_iou", ALPR_CONFIG)
        self.assertIn("save_crops", ALPR_CONFIG)
        self.assertIsInstance(ALPR_CONFIG["save_crops"], bool)

    def test_create_directories(self):
        """Verifica que create_directories funcione."""
        try:
            create_directories()
            self.assertTrue(DATA_DIR.exists())
            self.assertTrue(MODELS_DIR.exists())
            self.assertTrue(RESULTS_DIR.exists())
        except Exception as e:
            self.fail(f"create_directories() falló: {e}")

if __name__ == "__main__":
    unittest.main()
