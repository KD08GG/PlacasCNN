# PlacasCNN - Sistema ALPR (Automatic License Plate Recognition)

Sistema completo de reconocimiento automático de placas vehiculares usando YOLOv8 para detección, segmentación clásica de caracteres, y CNN para clasificación.

## 📋 Características

- **Detección de placas**: YOLOv8 para detectar placas en imágenes
- **Segmentación de caracteres**: Algoritmo clásico robusto (con opción a CNN)
- **Clasificación de caracteres**: CNN entrenada para reconocer 36 clases (0-9, A-Z)
- **Fallback OCR**: EasyOCR como respaldo cuando la CNN falla
- **Arquitectura modular**: Fácil de extender y mantener

## 📁 Estructura del Proyecto

```
PlacasCNN/
├── config.py                    # Configuración centralizada
├── requirements.txt             # Dependencias del proyecto
├── setup_dataset.py            # Utilidades para dataset
├── train_yolo.py               # Entrenamiento del detector
├── train_classifier.py         # Entrenamiento del clasificador
├── predict.py                  # Script de predicción
├── detectors/
│   └── plate_detector.py       # Detector de placas (YOLOv8)
├── segmenters/
│   ├── classical_segmenter.py  # Segmentador clásico
│   └── cnn_segmenter.py        # Segmentador CNN (placeholder)
├── recognizers/
│   ├── cnn_classifier.py       # Clasificador CNN de caracteres
│   └── easyocr_fallback.py     # Fallback con EasyOCR
├── pipeline/
│   └── alpr_pipeline.py        # Pipeline completo de ALPR
└── utils/
    ├── image_utils.py          # Utilidades de imagen
    └── plate_format.py         # Validación de formato
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip
- (Opcional) GPU con CUDA para entrenamiento más rápido

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repo>
cd PlacasCNN
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 📊 Preparación del Dataset

### 1. Dataset de Detección (YOLOv8)

Necesitas un dataset en formato Roboflow con:
- Carpetas: `train/`, `valid/`, `test/`
- Cada carpeta con subcarpetas `images/` y `labels/`
- Archivo `data.yaml` con configuración

**Descomprimir dataset:**

```python
from setup_dataset import unzip_dataset, verify_dataset_structure
from config import create_directories

create_directories()
unzip_dataset("path/to/your/dataset.zip")
verify_dataset_structure()
```

### 2. Dataset de Caracteres (Clasificador)

Estructura necesaria:
```
data/char_dataset/
├── 0/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── 1/
├── 2/
├── ...
├── A/
├── B/
└── Z/
```

Cada carpeta contiene imágenes del caracter correspondiente.

## 🎯 Entrenamiento

### 1. Entrenar Detector YOLOv8

```bash
python train_yolo.py --data data/dataset/data.yaml
```

**Parámetros personalizables en `config.py`:**
- `epochs`: Número de épocas (default: 50)
- `imgsz`: Tamaño de imagen (default: 640)
- `batch`: Tamaño de batch (default: 8)
- `device`: GPU/CPU (default: auto-detect)

### 2. Entrenar Clasificador de Caracteres

```bash
python train_classifier.py --data data/char_dataset --epochs 50
```

El modelo se guardará en `models/classifier/classifier.h5`

**Arquitectura del clasificador:**
- Input: 32x32 grayscale
- 3 bloques Conv2D + MaxPool
- GlobalAveragePooling
- Dense(128) + Dense(36)
- Activación: softmax

## 🔮 Predicción

### Uso Básico

```bash
python predict.py --image path/to/image.jpg
```

### Opciones Avanzadas

```bash
# Sin fallback de EasyOCR
python predict.py --image path/to/image.jpg --no-easyocr

# Con modelo clasificador personalizado
python predict.py --image path/to/image.jpg --classifier path/to/model.h5
```

### Uso Programático

```python
from pipeline.alpr_pipeline import ALPRPipeline
from config import CLASSIFIER_MODEL_DIR
from pathlib import Path

# Inicializar pipeline
classifier_path = Path(CLASSIFIER_MODEL_DIR) / "classifier.h5"
pipeline = ALPRPipeline(
    classifier_model=str(classifier_path),
    use_easyocr=True,
    segmenter_type="classical"
)

# Procesar imagen
results = pipeline.recognize_from_path(
    "path/to/image.jpg",
    save_crops=True,
    visualize=True
)

# Resultados
for result in results:
    print(f"Placa: {result['plate']}")
    print(f"Método: {result['method']}")
    print(f"Confianza: {result['conf']:.2f}")
```

## ⚙️ Configuración

Edita `config.py` para personalizar:

### Rutas
- `DATA_DIR`: Directorio de datos
- `MODELS_DIR`: Directorio de modelos
- `RESULTS_DIR`: Directorio de resultados

### Parámetros de YOLO
```python
YOLO_TRAIN_CONFIG = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 8,
    "device": None,
}
```

### Parámetros del Clasificador
```python
CLASSIFIER_CONFIG = {
    "img_size": 32,
    "batch": 64,
    "epochs": 50,
    "num_classes": 36,
}
```

### Parámetros del Pipeline
```python
ALPR_CONFIG = {
    "yolo_conf": 0.4,        # Umbral de confianza
    "yolo_iou": 0.45,        # Umbral IoU
    "save_crops": True,      # Guardar recortes
    "use_easyocr_fallback": True,  # Usar fallback
}
```

## 🔧 Módulos Principales

### PlateDetector
Detecta placas en imágenes usando YOLOv8.

```python
from detectors.plate_detector import PlateDetector

detector = PlateDetector()
crops = detector.detect(image, conf=0.4, iou=0.45)
```

### ClassicalSegmenter
Segmenta caracteres usando procesamiento de imagen clásico.

```python
from segmenters.classical_segmenter import ClassicalSegmenter

segmenter = ClassicalSegmenter()
chars = segmenter.segment(plate_image)
```

### CharacterClassifier
Clasifica caracteres individuales.

```python
from recognizers.cnn_classifier import CharacterClassifier

classifier = CharacterClassifier("path/to/model.h5")
char, confidence = classifier.classify(char_image)
```

### ALPRPipeline
Pipeline completo que orquesta todos los componentes.

```python
from pipeline.alpr_pipeline import ALPRPipeline

pipeline = ALPRPipeline(
    yolo_model=None,           # Usa modelo entrenado
    classifier_model="path/to/classifier.h5",
    use_easyocr=True,
    segmenter_type="classical"
)

results = pipeline.recognize_from_path("image.jpg")
```

## 📈 Flujo de Trabajo Completo

1. **Preparar Dataset**
   - Obtener dataset de placas (Roboflow)
   - Descomprimir con `setup_dataset.py`
   - Preparar dataset de caracteres

2. **Entrenar Modelos**
   - Entrenar YOLOv8: `python train_yolo.py`
   - Entrenar clasificador: `python train_classifier.py`

3. **Evaluar**
   - Probar con imágenes individuales
   - Ajustar parámetros en `config.py`

4. **Producción**
   - Usar `ALPRPipeline` en tu aplicación
   - Considerar optimizaciones (TFLite, ONNX)

## 🎨 Personalización

### Cambiar Formato de Placa

Edita `utils/plate_format.py`:

```python
# Ejemplo para formato mexicano ABC-123-D
PLATE_REGEX = re.compile(r'^[A-Z]{3}[-\s]?\d{3}[-\s]?[A-Z]$')

def is_valid_plate(s):
    # Tu lógica de validación
    pass
```

### Agregar Más Clases al Clasificador

Modifica `config.py`:

```python
CLASSIFIER_CONFIG = {
    "num_classes": 38,  # Agregar más caracteres
    "class_map": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-·")
}
```

### Usar Segmentador CNN

1. Entrenar modelo de segmentación
2. Implementar método `segment()` en `cnn_segmenter.py`
3. Usar en pipeline:

```python
pipeline = ALPRPipeline(
    segmenter_type="cnn",
    segmenter_model="path/to/segmenter.h5"
)
```

## 🐛 Solución de Problemas

### Error: "No se encontró data.yaml"
- Verifica que el dataset esté descomprimido correctamente
- Asegúrate de que `data.yaml` esté en la raíz del dataset

### Error: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Baja precisión en detección
- Aumentar epochs de entrenamiento
- Obtener más datos de entrenamiento
- Ajustar `yolo_conf` en `config.py`

### Segmentación no encuentra caracteres
- La imagen puede estar mal orientada
- Ajustar parámetros en `ClassicalSegmenter`
- Considerar preprocesamiento adicional

### Clasificador confunde caracteres similares
- Aumentar datos de entrenamiento
- Aumentar epochs
- Considerar data augmentation más agresivo

## 📝 TODOs / Mejoras Futuras

- [ ] Implementar segmentador CNN (UNet)
- [ ] Script para extraer caracteres automáticamente
- [ ] Sistema de evaluación (mAP, accuracy)
- [ ] API REST con FastAPI
- [ ] Soporte para video en tiempo real
- [ ] Optimización con TensorRT/ONNX
- [ ] Docker container
- [ ] Tests unitarios

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea tu rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

## 👥 Autores

- Tu nombre - [GitHub Profile]

## 🙏 Agradecimientos

- Ultralytics por YOLOv8
- Roboflow por facilitar datasets
- JaidedAI por EasyOCR
- TensorFlow/Keras team