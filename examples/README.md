# Ejemplos

Este directorio contiene ejemplos y archivos de prueba para el sistema ALPR.

## Estructura

```
examples/
├── README.md                    # Este archivo
├── ground_truth_example.json   # Ejemplo de archivo ground truth
└── test_image.jpg              # (Coloca tus imágenes aquí)
```

## Uso

### 1. Predicción Simple

Coloca una imagen de prueba en este directorio y ejecuta:

```bash
python predict.py --image examples/tu_imagen.jpg
```

Los resultados se guardarán en `results/crops/`

### 2. Evaluación con Ground Truth

Para evaluar el sistema, necesitas un archivo con las placas reales. Formato:

**JSON (`ground_truth.json`):**
```json
{
  "imagen1.jpg": "ABC123D",
  "imagen2.jpg": "XYZ789E",
  "imagen3.jpg": "LMN456P"
}
```

**O CSV (`ground_truth.csv`):**
```csv
image_name,plate_text
imagen1.jpg,ABC123D
imagen2.jpg,XYZ789E
imagen3.jpg,LMN456P
```

Luego ejecuta:

```bash
python evaluate.py --images examples/ --ground-truth examples/ground_truth.json
```

### 3. Procesar Múltiples Imágenes

Coloca todas tus imágenes en esta carpeta y usa un script:

```python
from pathlib import Path
from pipeline.alpr_pipeline import ALPRPipeline

pipeline = ALPRPipeline(
    classifier_model="models/classifier/classifier.h5",
    use_easyocr=True
)

images_dir = Path("examples")
for img_path in images_dir.glob("*.jpg"):
    results = pipeline.recognize_from_path(img_path)
    for r in results:
        print(f"{img_path.name}: {r['plate']}")
```

## Archivos de Ejemplo Incluidos

### ground_truth_example.json

Archivo de ejemplo mostrando el formato esperado para evaluación.

## Descarga de Imágenes de Prueba

Puedes obtener imágenes de prueba de:
- Tu propio conjunto de datos
- Datasets públicos de placas (con licencia apropiada)
- Generar sintéticas para testing

## Notas

- Las imágenes deben estar en formato JPG, JPEG o PNG
- Para mejores resultados, asegúrate de que las placas sean visibles
- El sistema funciona mejor con imágenes de buena calidad
