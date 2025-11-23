# tests/test_utils.py
"""Tests para utilidades."""
import unittest
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import resize_and_pad
from utils.plate_format import is_valid_plate, PLATE_REGEX

class TestImageUtils(unittest.TestCase):
    """Tests para utilidades de imagen."""

    def test_resize_and_pad_square(self):
        """Test resize_and_pad con imagen cuadrada."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = resize_and_pad(img, 200)
        self.assertEqual(result.shape, (200, 200))

    def test_resize_and_pad_rectangular(self):
        """Test resize_and_pad con imagen rectangular."""
        img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        result = resize_and_pad(img, 150)
        self.assertEqual(result.shape, (150, 150))

    def test_resize_and_pad_color(self):
        """Test resize_and_pad con imagen a color."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = resize_and_pad(img, 200)
        self.assertEqual(result.shape, (200, 200, 3))

class TestPlateFormat(unittest.TestCase):
    """Tests para validación de formato de placa."""

    def test_valid_plate_format(self):
        """Test con formato válido."""
        # Formato: AAA999A
        self.assertTrue(is_valid_plate("ABC123D"))
        self.assertTrue(is_valid_plate("XYZ789E"))

    def test_valid_plate_with_separators(self):
        """Test con separadores."""
        self.assertTrue(is_valid_plate("ABC-123-D"))
        self.assertTrue(is_valid_plate("ABC 123 D"))

    def test_invalid_plate_format(self):
        """Test con formatos inválidos."""
        self.assertFalse(is_valid_plate("12345"))
        self.assertFalse(is_valid_plate("ABCDEFG"))
        self.assertFalse(is_valid_plate(""))
        self.assertFalse(is_valid_plate(None))

    def test_plate_regex(self):
        """Test del regex de placa."""
        self.assertIsNotNone(PLATE_REGEX)
        self.assertTrue(PLATE_REGEX.match("ABC-123-D"))
        self.assertTrue(PLATE_REGEX.match("ABC 123 D"))

if __name__ == "__main__":
    unittest.main()
