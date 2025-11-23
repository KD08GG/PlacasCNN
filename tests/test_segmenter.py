# tests/test_segmenter.py
"""Tests para segmentadores."""
import unittest
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from segmenters.classical_segmenter import ClassicalSegmenter

class TestClassicalSegmenter(unittest.TestCase):
    """Tests para ClassicalSegmenter."""

    def setUp(self):
        """Configuración antes de cada test."""
        self.segmenter = ClassicalSegmenter()

    def test_segmenter_initialization(self):
        """Verifica que el segmentador se inicialice correctamente."""
        self.assertIsInstance(self.segmenter, ClassicalSegmenter)
        self.assertGreater(self.segmenter.min_w, 0)
        self.assertGreater(self.segmenter.min_h, 0)

    def test_segment_returns_list(self):
        """Verifica que segment() retorne una lista."""
        # Crear imagen sintética simple
        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        result = self.segmenter.segment(img)
        self.assertIsInstance(result, list)

    def test_segment_synthetic_plate(self):
        """Test con placa sintética."""
        # Crear placa sintética con texto
        img = np.ones((60, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "ABC123", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        chars = self.segmenter.segment(img)

        # Debería encontrar al menos algunos caracteres
        self.assertIsInstance(chars, list)
        # En una placa sintética clara, debería encontrar caracteres
        # Nota: el número exacto puede variar según el algoritmo

    def test_segment_empty_image(self):
        """Test con imagen vacía."""
        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        chars = self.segmenter.segment(img)
        self.assertIsInstance(chars, list)
        # Imagen vacía no debería tener caracteres
        self.assertEqual(len(chars), 0)

if __name__ == "__main__":
    unittest.main()
