# segmenters/cnn_segmenter.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils.image_utils import resize_and_pad

class CNNSegmenter:
    """
    Cargador para segmenter CNN (unet-like). Debes entrenar y guardar el modelo en SEGMENTER_MODEL_DIR.
    Método segment debe devolver lista de recortes (numpy images).
    """
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def segment(self, plate_img):
        # Preprocess to model input size, predict mask, then extract connected components
        # Implementación depende del modelo concreto. Aquí devolvemos [] por defecto.
        raise NotImplementedError("Entrena y guarda un modelo de segmentación para usar CNNSegmenter.")
