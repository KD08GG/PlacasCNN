# pipeline/alpr_pipeline.py
from detectors.plate_detector import PlateDetector
from segmenters.classical_segmenter import ClassicalSegmenter
from recognizers.cnn_classifier import CharacterClassifier
from recognizers.easyocr_fallback import EasyOCRFallback
from config import ALPR_CONFIG, CLASSIFIER_CONFIG, CROPS_DIR
import cv2
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ALPRPipeline:
    def __init__(self, yolo_model=None, classifier_model=None, use_easyocr=True, segmenter_type="classical", segmenter_model=None):
        """
        Inicializa el pipeline completo de ALPR.

        Args:
            yolo_model: Ruta al modelo YOLO (None para usar el mejor entrenado)
            classifier_model: Ruta al modelo clasificador CNN
            use_easyocr: Si usar EasyOCR como fallback
            segmenter_type: 'classical' o 'cnn'
            segmenter_model: Ruta al modelo segmentador (si type='cnn')
        """
        try:
            logger.info("Inicializando ALPR Pipeline...")
            self.detector = PlateDetector(yolo_model)
            logger.info("✓ Detector cargado")

            # segmenter_type: 'classical' or 'cnn'
            if segmenter_type == "classical":
                self.segmenter = ClassicalSegmenter()
                logger.info("✓ Segmentador clásico inicializado")
            else:
                from segmenters.cnn_segmenter import CNNSegmenter
                self.segmenter = CNNSegmenter(segmenter_model)
                logger.info("✓ Segmentador CNN cargado")

            if classifier_model and Path(classifier_model).exists():
                self.classifier = CharacterClassifier(classifier_model)
                logger.info("✓ Clasificador CNN cargado")
            else:
                logger.warning(f"⚠ Modelo clasificador no encontrado: {classifier_model}")
                self.classifier = None

            if use_easyocr:
                self.fallback = EasyOCRFallback()
                logger.info("✓ EasyOCR fallback habilitado")
            else:
                self.fallback = None
                logger.info("EasyOCR fallback deshabilitado")

            Path(CROPS_DIR).mkdir(parents=True, exist_ok=True)
            logger.info("Pipeline inicializado correctamente\n")

        except Exception as e:
            logger.error(f"Error inicializando pipeline: {e}")
            raise

    def recognize_from_path(self, image_path, save_crops=True, visualize=False):
        """
        Procesa una imagen y reconoce las placas.

        Args:
            image_path: Ruta a la imagen
            save_crops: Si guardar recortes de placas y caracteres
            visualize: Si imprimir resultados en consola

        Returns:
            Lista de diccionarios con resultados
        """
        try:
            # Cargar imagen
            img = cv2.imread(str(image_path))
            if img is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

            # Detectar placas
            try:
                plates = self.detector.detect(img, conf=ALPR_CONFIG["yolo_conf"], iou=ALPR_CONFIG["yolo_iou"])
                logger.info(f"Detectadas {len(plates)} placa(s)")
            except Exception as e:
                logger.error(f"Error en detección: {e}")
                return []

            outputs = []

            for i, p in enumerate(plates):
                try:
                    plate_img = p["crop"]

                    # Segmentar caracteres
                    try:
                        chars = self.segmenter.segment(plate_img)
                        logger.debug(f"Placa {i}: {len(chars)} caracteres segmentados")
                    except Exception as e:
                        logger.warning(f"Error en segmentación de placa {i}: {e}")
                        chars = []

                    # Si no hay caracteres segmentados, usar fallback
                    if not chars:
                        if self.fallback:
                            logger.info(f"Placa {i}: Usando EasyOCR fallback")
                            try:
                                text = self.fallback.read_plate(plate_img)
                                outputs.append({
                                    "plate": text or "UNKNOWN",
                                    "method": "easyocr_fallback",
                                    "conf": 0.0
                                })
                                continue
                            except Exception as e:
                                logger.error(f"Error en EasyOCR: {e}")
                                outputs.append({
                                    "plate": "ERROR",
                                    "method": "error",
                                    "conf": 0.0
                                })
                                continue
                        else:
                            logger.warning(f"Placa {i}: No se pudieron segmentar caracteres y no hay fallback")
                            outputs.append({
                                "plate": "UNKNOWN",
                                "method": "no_chars",
                                "conf": 0.0
                            })
                            continue

                    # Clasificar caracteres
                    predicted = []
                    confidences = []

                    if self.classifier:
                        for ch_idx, ch in enumerate(chars):
                            try:
                                c, conf = self.classifier.classify(ch)
                                predicted.append(c)
                                confidences.append(conf)
                            except Exception as e:
                                logger.warning(f"Error clasificando carácter {ch_idx}: {e}")
                                predicted.append("?")
                                confidences.append(0.0)
                    else:
                        # Si no hay clasificador, usar fallback
                        if self.fallback:
                            logger.info(f"Placa {i}: Sin clasificador, usando EasyOCR")
                            try:
                                text = self.fallback.read_plate(plate_img)
                                outputs.append({
                                    "plate": text or "UNKNOWN",
                                    "method": "easyocr_no_classifier",
                                    "conf": 0.0
                                })
                                continue
                            except Exception as e:
                                logger.error(f"Error en EasyOCR: {e}")

                    text = "".join(predicted) if predicted else "UNKNOWN"
                    method = "cnn_chars" if predicted else "no_classification"

                    # Guardar crops si se solicita
                    if save_crops:
                        try:
                            crop_dir = Path(CROPS_DIR)
                            crop_dir.mkdir(parents=True, exist_ok=True)

                            plate_path = crop_dir / f"{Path(image_path).stem}_plate_{i}.jpg"
                            cv2.imwrite(str(plate_path), plate_img)

                            for j, ch in enumerate(chars):
                                char_path = crop_dir / f"{Path(image_path).stem}_plate_{i}_char_{j}.png"
                                cv2.imwrite(str(char_path), ch)
                        except Exception as e:
                            logger.warning(f"Error guardando crops: {e}")

                    avg_conf = float(sum(confidences)/len(confidences)) if confidences else 0.0

                    outputs.append({
                        "plate": text,
                        "method": method,
                        "conf": avg_conf
                    })

                    if visualize:
                        print(f"Placa {i}: {text} (method={method}, conf={avg_conf:.2f})")

                except Exception as e:
                    logger.error(f"Error procesando placa {i}: {e}")
                    outputs.append({
                        "plate": "ERROR",
                        "method": "error",
                        "conf": 0.0
                    })

            return outputs

        except Exception as e:
            logger.error(f"Error en recognize_from_path: {e}")
            return []
