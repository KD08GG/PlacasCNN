# evaluate.py
"""
Sistema de evaluación para el pipeline ALPR.
Calcula métricas de detección, segmentación y reconocimiento.
"""
import cv2
import numpy as np
from pathlib import Path
from pipeline.alpr_pipeline import ALPRPipeline
from config import create_directories, CLASSIFIER_MODEL_DIR, RESULTS_DIR
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

class ALPREvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = []

    def load_ground_truth(self, gt_file):
        """
        Carga archivo de ground truth.

        Formato esperado (JSON):
        {
            "image1.jpg": "ABC123D",
            "image2.jpg": "XYZ789E",
            ...
        }

        O formato CSV:
        image_name,plate_text
        image1.jpg,ABC123D
        image2.jpg,XYZ789E
        """
        gt_file = Path(gt_file)
        if gt_file.suffix == ".json":
            with open(gt_file, 'r') as f:
                return json.load(f)
        elif gt_file.suffix == ".csv":
            df = pd.read_csv(gt_file)
            return dict(zip(df['image_name'], df['plate_text']))
        else:
            raise ValueError("Formato no soportado. Use .json o .csv")

    def normalize_plate(self, text):
        """Normaliza texto de placa (mayúsculas, sin espacios)."""
        if not text:
            return ""
        return text.upper().replace(" ", "").replace("-", "")

    def calculate_char_accuracy(self, predicted, ground_truth):
        """Calcula accuracy a nivel de carácter."""
        pred = self.normalize_plate(predicted)
        gt = self.normalize_plate(ground_truth)

        if not gt:
            return 0.0

        # Alinear strings (usar la más corta)
        min_len = min(len(pred), len(gt))
        if min_len == 0:
            return 0.0

        matches = sum(1 for i in range(min_len) if pred[i] == gt[i])
        return matches / len(gt)

    def calculate_levenshtein(self, s1, s2):
        """Calcula distancia de Levenshtein."""
        if len(s1) < len(s2):
            return self.calculate_levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def evaluate_image(self, image_path, ground_truth):
        """Evalúa una imagen individual."""
        try:
            results = self.pipeline.recognize_from_path(
                image_path,
                save_crops=False,
                visualize=False
            )

            if not results:
                return {
                    "image": str(image_path),
                    "ground_truth": ground_truth,
                    "predicted": "",
                    "detected": False,
                    "exact_match": False,
                    "char_accuracy": 0.0,
                    "levenshtein": len(ground_truth),
                    "confidence": 0.0,
                    "method": "none"
                }

            # Usar el primer resultado (puede extenderse para múltiples placas)
            result = results[0]
            predicted = result["plate"]
            confidence = result["conf"]
            method = result["method"]

            # Normalizar
            pred_norm = self.normalize_plate(predicted)
            gt_norm = self.normalize_plate(ground_truth)

            # Métricas
            exact_match = pred_norm == gt_norm
            char_acc = self.calculate_char_accuracy(predicted, ground_truth)
            lev_dist = self.calculate_levenshtein(pred_norm, gt_norm)

            return {
                "image": str(Path(image_path).name),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "detected": True,
                "exact_match": exact_match,
                "char_accuracy": char_acc,
                "levenshtein": lev_dist,
                "confidence": confidence,
                "method": method
            }

        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return {
                "image": str(image_path),
                "ground_truth": ground_truth,
                "predicted": "",
                "detected": False,
                "exact_match": False,
                "char_accuracy": 0.0,
                "levenshtein": len(ground_truth),
                "confidence": 0.0,
                "method": "error",
                "error": str(e)
            }

    def evaluate_dataset(self, images_dir, ground_truth):
        """Evalúa un conjunto de imágenes."""
        images_dir = Path(images_dir)
        self.results = []

        print(f"Evaluando {len(ground_truth)} imágenes...")

        for img_name, gt_text in tqdm(ground_truth.items(), desc="Evaluando"):
            img_path = images_dir / img_name
            if not img_path.exists():
                print(f"Advertencia: {img_path} no encontrada")
                continue

            result = self.evaluate_image(img_path, gt_text)
            self.results.append(result)

        return self.compute_metrics()

    def compute_metrics(self):
        """Calcula métricas agregadas."""
        if not self.results:
            return {}

        total = len(self.results)
        detected = sum(1 for r in self.results if r["detected"])
        exact_matches = sum(1 for r in self.results if r["exact_match"])

        char_accuracies = [r["char_accuracy"] for r in self.results]
        confidences = [r["confidence"] for r in self.results if r["detected"]]

        metrics = {
            "total_images": total,
            "detected_plates": detected,
            "detection_rate": detected / total if total > 0 else 0,
            "exact_match_accuracy": exact_matches / total if total > 0 else 0,
            "mean_char_accuracy": np.mean(char_accuracies),
            "median_char_accuracy": np.median(char_accuracies),
            "mean_confidence": np.mean(confidences) if confidences else 0,
            "method_distribution": self._count_methods()
        }

        return metrics

    def _count_methods(self):
        """Cuenta distribución de métodos usados."""
        methods = {}
        for r in self.results:
            method = r.get("method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        return methods

    def save_results(self, output_dir):
        """Guarda resultados en archivos."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar resultados detallados (CSV)
        df = pd.DataFrame(self.results)
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Resultados guardados en: {csv_path}")

        # Guardar métricas (JSON)
        metrics = self.compute_metrics()
        json_path = output_dir / f"evaluation_metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Métricas guardadas en: {json_path}")

        # Generar reporte
        self.generate_report(output_dir, timestamp, metrics)

        return csv_path, json_path

    def generate_report(self, output_dir, timestamp, metrics):
        """Genera reporte en texto."""
        report_path = output_dir / f"evaluation_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE EVALUACIÓN - Sistema ALPR\n")
            f.write("="*60 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("MÉTRICAS GENERALES\n")
            f.write("-"*60 + "\n")
            f.write(f"Total de imágenes:           {metrics['total_images']}\n")
            f.write(f"Placas detectadas:           {metrics['detected_plates']}\n")
            f.write(f"Tasa de detección:           {metrics['detection_rate']:.2%}\n")
            f.write(f"Accuracy (coincidencia exacta): {metrics['exact_match_accuracy']:.2%}\n")
            f.write(f"Accuracy promedio (caracteres): {metrics['mean_char_accuracy']:.2%}\n")
            f.write(f"Accuracy mediana (caracteres):  {metrics['median_char_accuracy']:.2%}\n")
            f.write(f"Confianza promedio:          {metrics['mean_confidence']:.2%}\n\n")

            f.write("DISTRIBUCIÓN DE MÉTODOS\n")
            f.write("-"*60 + "\n")
            for method, count in metrics['method_distribution'].items():
                pct = count / metrics['total_images'] * 100
                f.write(f"{method:20s}: {count:4d} ({pct:5.1f}%)\n")

            f.write("\n" + "="*60 + "\n")

        print(f"✓ Reporte generado en: {report_path}")

    def print_summary(self):
        """Imprime resumen en consola."""
        metrics = self.compute_metrics()

        print("\n" + "="*60)
        print("RESUMEN DE EVALUACIÓN")
        print("="*60)
        print(f"Total de imágenes:              {metrics['total_images']}")
        print(f"Placas detectadas:              {metrics['detected_plates']}")
        print(f"Tasa de detección:              {metrics['detection_rate']:.2%}")
        print(f"Accuracy (coincidencia exacta): {metrics['exact_match_accuracy']:.2%}")
        print(f"Accuracy promedio (caracteres): {metrics['mean_char_accuracy']:.2%}")
        print(f"Confianza promedio:             {metrics['mean_confidence']:.2%}")
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluar sistema ALPR")
    parser.add_argument("--images", required=True, help="Directorio con imágenes de test")
    parser.add_argument("--ground-truth", required=True, help="Archivo con ground truth (.json o .csv)")
    parser.add_argument("--classifier", help="Ruta a modelo clasificador")
    parser.add_argument("--output", default=None, help="Directorio de salida para resultados")
    parser.add_argument("--no-easyocr", action="store_true", help="Desactivar EasyOCR")

    args = parser.parse_args()

    create_directories()

    # Cargar clasificador
    classifier_path = args.classifier or str(Path(CLASSIFIER_MODEL_DIR) / "classifier.h5")

    # Inicializar pipeline
    print("Inicializando pipeline...")
    pipeline = ALPRPipeline(
        classifier_model=classifier_path,
        use_easyocr=not args.no_easyocr
    )

    # Inicializar evaluador
    evaluator = ALPREvaluator(pipeline)

    # Cargar ground truth
    print(f"Cargando ground truth desde {args.ground_truth}...")
    ground_truth = evaluator.load_ground_truth(args.ground_truth)
    print(f"✓ {len(ground_truth)} anotaciones cargadas")

    # Evaluar
    metrics = evaluator.evaluate_dataset(args.images, ground_truth)

    # Mostrar resumen
    evaluator.print_summary()

    # Guardar resultados
    output_dir = args.output or (Path(RESULTS_DIR) / "evaluation")
    evaluator.save_results(output_dir)

if __name__ == "__main__":
    main()
