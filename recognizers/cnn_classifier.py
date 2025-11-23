# recognizers/cnn_classifier.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from config import CLASSIFIER_CONFIG

CLASS_MAP = CLASSIFIER_CONFIG["class_map"]

def preprocess_char(img, img_size=32):
    import cv2
    # convertir a gray y escalar a img_size
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    # pad to square
    size = max(h,w)
    top = (size - h)//2
    left = (size - w)//2
    canvas = 255 * np.ones((size,size), dtype=img.dtype)
    canvas[top:top+h, left:left+w] = img
    resized = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # normalize 0-1
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, -1)  # (H,W,1)
    return x

class CharacterClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.img_size = CLASSIFIER_CONFIG["img_size"]

    def classify(self, img):
        x = preprocess_char(img, img_size=self.img_size)
        pred = self.model.predict(np.expand_dims(x, 0))[0]
        idx = int(np.argmax(pred))
        char = CLASS_MAP[idx]
        conf = float(pred[idx])
        return char, conf
