# alpr.py - Sistema ALPR simplificado (TODO EN UNO)
"""
Sistema completo de reconocimiento de placas en un solo archivo.
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración simple del sistema."""
    # Directorios
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # YOLOv8
    YOLO_MODEL = "yolov8n.pt"  # Modelo por defecto
    YOLO_CONF = 0.4
    YOLO_IOU = 0.45

    # Clasificador
    IMG_SIZE = 32
    NUM_CLASSES = 36
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    @staticmethod
    def setup():
        """Crea directorios necesarios."""
        for d in [Config.DATA_DIR, Config.MODELS_DIR, Config.RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DETECTOR DE PLACAS
# ============================================================================

class PlateDetector:
    """Detecta placas usando YOLOv8."""

    def __init__(self, model_path=None):
        model_path = model_path or Config.YOLO_MODEL
        self.model = YOLO(model_path)

    def detect(self, image):
        """Detecta placas y retorna lista de crops."""
        results = self.model(image, conf=Config.YOLO_CONF, iou=Config.YOLO_IOU)
        boxes = results[0].boxes

        plates = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2].copy()
            plates.append({
                'crop': crop,
                'bbox': (x1, y1, x2, y2),
                'conf': float(box.conf[0])
            })
        return plates

# ============================================================================
# SEGMENTADOR DE CARACTERES
# ============================================================================

class CharSegmenter:
    """Segmenta caracteres de una placa."""

    def segment(self, plate_img):
        """Segmenta caracteres usando procesamiento clásico."""
        # Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img

        # Redimensionar para consistencia
        h = 100
        scale = h / gray.shape[0]
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), h))

        # Threshold adaptativo
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 10)

        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar y ordenar por posición X
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 6 and h > 12:  # Filtro de tamaño mínimo
                # Mapear de vuelta a imagen original
                x0 = int(x / scale)
                y0 = int(y / scale)
                w0 = int(w / scale)
                h0 = int(h / scale)
                boxes.append((x0, y0, w0, h0))

        # Ordenar por X
        boxes.sort(key=lambda b: b[0])

        # Extraer caracteres
        chars = []
        for x, y, w, h in boxes:
            char_img = plate_img[y:y+h, x:x+w]
            chars.append(char_img)

        return chars

# ============================================================================
# CLASIFICADOR CNN
# ============================================================================

class CharClassifier:
    """Clasifica caracteres individuales."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path and Path(model_path).exists():
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)

    def preprocess(self, img):
        """Preprocesa imagen de carácter."""
        # Convertir a gris
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Hacer cuadrado con padding
        h, w = img.shape
        size = max(h, w)
        canvas = np.ones((size, size), dtype=img.dtype) * 255
        top = (size - h) // 2
        left = (size - w) // 2
        canvas[top:top+h, left:left+w] = img

        # Redimensionar
        resized = cv2.resize(canvas, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Normalizar
        normalized = resized.astype('float32') / 255.0
        return np.expand_dims(normalized, -1)

    def classify(self, img):
        """Clasifica un carácter."""
        if self.model is None:
            return '?', 0.0

        x = self.preprocess(img)
        pred = self.model.predict(np.expand_dims(x, 0), verbose=0)[0]
        idx = int(np.argmax(pred))
        char = Config.CHARS[idx]
        conf = float(pred[idx])
        return char, conf

# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

class ALPRSystem:
    """Sistema completo de ALPR."""

    def __init__(self, yolo_model=None, classifier_model=None, use_ocr_fallback=True):
        """
        Inicializa el sistema ALPR.

        Args:
            yolo_model: Ruta al modelo YOLO (None = usar default)
            classifier_model: Ruta al clasificador CNN (None = solo OCR)
            use_ocr_fallback: Usar EasyOCR como fallback
        """
        print("Inicializando sistema ALPR...")

        self.detector = PlateDetector(yolo_model)
        self.segmenter = CharSegmenter()
        self.classifier = CharClassifier(classifier_model)

        self.ocr = None
        if use_ocr_fallback:
            print("Cargando EasyOCR...")
            self.ocr = easyocr.Reader(['en'], gpu=False)

        print("Sistema listo!\n")

    def recognize(self, image_path):
        """
        Reconoce placas en una imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Lista de placas detectadas con texto
        """
        # Cargar imagen
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: No se pudo cargar {image_path}")
            return []

        # Detectar placas
        plates = self.detector.detect(img)
        print(f"Detectadas {len(plates)} placa(s)")

        results = []
        for i, plate_data in enumerate(plates):
            plate_img = plate_data['crop']

            # Segmentar caracteres
            chars = self.segmenter.segment(plate_img)

            # Intentar clasificar con CNN
            text = ""
            method = "unknown"

            if chars and self.classifier.model:
                # Clasificar cada carácter
                predicted = []
                for char_img in chars:
                    char, conf = self.classifier.classify(char_img)
                    predicted.append(char)
                text = "".join(predicted)
                method = "cnn"

            # Fallback a OCR si no hay texto
            if not text and self.ocr:
                try:
                    ocr_result = self.ocr.readtext(plate_img, detail=0)
                    text = "".join(ocr_result).upper() if ocr_result else "UNKNOWN"
                    method = "ocr"
                except:
                    text = "UNKNOWN"
                    method = "error"

            results.append({
                'plate': text,
                'confidence': plate_data['conf'],
                'method': method,
                'bbox': plate_data['bbox']
            })

            print(f"  Placa {i+1}: {text} (método: {method})")

        return results

    def batch_recognize(self, images_dir):
        """Procesa múltiples imágenes."""
        images_dir = Path(images_dir)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        print(f"\nProcesando {len(image_files)} imágenes...\n")

        all_results = []
        for img_path in image_files:
            print(f"=== {img_path.name} ===")
            results = self.recognize(img_path)
            all_results.append({
                'image': img_path.name,
                'plates': results
            })
            print()

        return all_results

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sistema ALPR simplificado")
    parser.add_argument("--image", help="Imagen a procesar")
    parser.add_argument("--dir", help="Directorio con imágenes")
    parser.add_argument("--yolo", help="Modelo YOLO custom")
    parser.add_argument("--classifier", help="Modelo clasificador CNN")
    parser.add_argument("--no-ocr", action="store_true", help="Desactivar OCR fallback")

    args = parser.parse_args()

    # Setup
    Config.setup()

    # Inicializar sistema
    system = ALPRSystem(
        yolo_model=args.yolo,
        classifier_model=args.classifier,
        use_ocr_fallback=not args.no_ocr
    )

    # Procesar
    if args.image:
        system.recognize(args.image)
    elif args.dir:
        system.batch_recognize(args.dir)
    else:
        print("Uso: python alpr.py --image foto.jpg")
        print("  o: python alpr.py --dir carpeta/")
