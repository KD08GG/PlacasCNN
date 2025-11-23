# pipeline/alpr_pipeline.py
from detectors.plate_detector import PlateDetector
from segmenters.classical_segmenter import ClassicalSegmenter
from recognizers.cnn_classifier import CharacterClassifier
from recognizers.easyocr_fallback import EasyOCRFallback
from config import ALPR_CONFIG, CLASSIFIER_CONFIG, CROPS_DIR
import cv2
from pathlib import Path

class ALPRPipeline:
    def __init__(self, yolo_model=None, classifier_model=None, use_easyocr=True, segmenter_type="classical", segmenter_model=None):
        self.detector = PlateDetector(yolo_model)
        # segmenter_type: 'classical' or 'cnn'
        if segmenter_type == "classical":
            self.segmenter = ClassicalSegmenter()
        else:
            # if you implemented cnn segmenter
            from segmenters.cnn_segmenter import CNNSegmenter
            self.segmenter = CNNSegmenter(segmenter_model)
        self.classifier = CharacterClassifier(classifier_model)
        self.fallback = EasyOCRFallback() if use_easyocr else None
        Path(CROPS_DIR).mkdir(parents=True, exist_ok=True)

    def recognize_from_path(self, image_path, save_crops=True, visualize=False):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        plates = self.detector.detect(img, conf=ALPR_CONFIG["yolo_conf"], iou=ALPR_CONFIG["yolo_iou"])
        outputs = []
        for i, p in enumerate(plates):
            plate_img = p["crop"]
            chars = self.segmenter.segment(plate_img)
            if not chars and self.fallback:
                text = self.fallback.read_plate(plate_img)
                outputs.append({"plate": text or "UNKNOWN", "method": "easyocr_fallback", "conf": 0.0})
                continue
            predicted = []
            confidences = []
            for ch in chars:
                c, conf = self.classifier.classify(ch)
                predicted.append(c)
                confidences.append(conf)
            text = "".join(predicted) if predicted else "UNKNOWN"
            method = "cnn_chars" if predicted else "UNKNOWN"
            if save_crops:
                cv2.imwrite(str(Path(CROPS_DIR) / f"{Path(image_path).stem}_plate_{i}.jpg"), plate_img)
                for j, ch in enumerate(chars):
                    cv2.imwrite(str(Path(CROPS_DIR) / f"{Path(image_path).stem}_plate_{i}_char_{j}.png"), ch)
            outputs.append({"plate": text, "method": method, "conf": float(sum(confidences)/len(confidences)) if confidences else 0.0})
            if visualize:
                print(f"Plate {i}: {text} (method={method}, conf={outputs[-1]['conf']:.2f})")
        return outputs
