# PlacasCNN - Sistema ALPR Simplificado

Sistema de reconocimiento de placas vehiculares **simple y directo**. Todo en pocos archivos.

## 📦 Archivos Principales

```
PlacasCNN/
├── alpr.py          # TODO EL SISTEMA EN UN ARCHIVO
├── train.py         # Entrenamiento simplificado
├── requirements.txt # Dependencias
└── README_SIMPLE.md # Este archivo
```

## 🚀 Inicio Rápido (3 pasos)

### 1. Instalar

```bash
pip install -r requirements.txt
```

### 2. Generar datos sintéticos y entrenar

```bash
# Generar caracteres sintéticos
python train.py synthetic --samples 100 --output data/chars

# Entrenar clasificador (rápido, 10 epochs)
python train.py classifier --data data/chars --epochs 10
```

### 3. Usar

```bash
# Procesar una imagen
python alpr.py --image foto.jpg

# Procesar carpeta
python alpr.py --dir carpeta_imagenes/
```

## 📖 Uso Detallado

### Reconocer Placas

```bash
# Imagen individual
python alpr.py --image mi_imagen.jpg

# Carpeta completa
python alpr.py --dir mis_imagenes/

# Con modelo YOLO personalizado
python alpr.py --image foto.jpg --yolo models/yolo/detector/weights/best.pt

# Con clasificador personalizado
python alpr.py --image foto.jpg --classifier models/classifier.h5

# Sin OCR fallback
python alpr.py --image foto.jpg --no-ocr
```

### Entrenar

```bash
# 1. Generar datos sintéticos
python train.py synthetic --samples 200 --output data/chars

# 2. Entrenar clasificador
python train.py classifier --data data/chars --epochs 30

# 3. Entrenar detector (necesitas dataset YOLO)
python train.py detector --data data/dataset/data.yaml --epochs 50
```

### Usar desde Python

```python
from alpr import ALPRSystem

# Inicializar
system = ALPRSystem(
    classifier_model="models/classifier.h5",
    use_ocr_fallback=True
)

# Reconocer
results = system.recognize("foto.jpg")

for r in results:
    print(f"Placa: {r['plate']}")
    print(f"Confianza: {r['confidence']:.2f}")
    print(f"Método: {r['method']}")
```

## 🏗️ Arquitectura Simplificada

### alpr.py - Un solo archivo con todo:

- **Config**: Configuración simple
- **PlateDetector**: Detecta placas con YOLOv8
- **CharSegmenter**: Segmenta caracteres (procesamiento clásico)
- **CharClassifier**: Clasifica caracteres con CNN
- **ALPRSystem**: Pipeline completo

### train.py - Entrenamiento:

- `train_detector()`: Entrena YOLOv8
- `train_classifier()`: Entrena CNN
- `generate_synthetic_data()`: Genera datos de prueba

## 🎯 Flujos de Trabajo

### Opción 1: Prueba Rápida (5 min)

```bash
# Generar + entrenar
python train.py synthetic --samples 50
python train.py classifier --data data/synthetic_chars --epochs 10

# Usar
python alpr.py --image test.jpg
```

### Opción 2: Con Dataset Real

```bash
# 1. Entrenar detector con tu dataset YOLO
python train.py detector --data tu_dataset/data.yaml

# 2. Preparar caracteres (manualmente o extracción)
# Estructura: data/chars/0/, data/chars/1/, ..., data/chars/Z/

# 3. Entrenar clasificador
python train.py classifier --data data/chars --epochs 50

# 4. Usar con modelos entrenados
python alpr.py --image foto.jpg \
  --yolo models/yolo/detector/weights/best.pt \
  --classifier models/classifier.h5
```

## ⚙️ Configuración

Edita las constantes en `alpr.py`:

```python
class Config:
    YOLO_CONF = 0.4      # Umbral de confianza YOLO
    YOLO_IOU = 0.45      # IoU para NMS
    IMG_SIZE = 32        # Tamaño de caracteres
    NUM_CLASSES = 36     # 0-9, A-Z
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

## 🔧 Dependencias

```
ultralytics    # YOLOv8
opencv-python  # Procesamiento de imagen
numpy          # Arrays
tensorflow     # CNN
easyocr        # OCR fallback
```

## 📝 Notas

- **Simple**: Todo en 2 archivos principales
- **Funcional**: Mismo resultado, menos complejidad
- **Extensible**: Fácil de modificar para tus necesidades
- **Sin dependencias extra**: Solo lo esencial

## 🐛 Solución de Problemas

### Error: "No module named 'ultralytics'"

```bash
pip install -r requirements.txt
```

### No detecta placas

- Verifica que la imagen sea clara
- Ajusta `YOLO_CONF` en `alpr.py` (probar 0.3 o 0.2)
- Entrena tu propio detector con tu dataset

### Clasificador no funciona

- Entrena el clasificador primero
- O usa solo OCR: `python alpr.py --image foto.jpg` (usa EasyOCR por default)

## 📈 Mejoras Posibles

Si necesitas más funcionalidad:

1. **Evaluación**: Agrega función para calcular métricas
2. **Validación**: Agrega regex para validar formato de placa
3. **API**: Envuelve en Flask/FastAPI
4. **Optimización**: Convierte a ONNX/TFLite

## 🆚 Versión Completa vs Simplificada

**Versión Completa** (archivos anteriores):
- ✅ Más modular
- ✅ Tests unitarios
- ✅ Evaluación completa
- ✅ Notebooks
- ❌ Muchos archivos
- ❌ Más complejo

**Versión Simplificada** (este):
- ✅ 2 archivos principales
- ✅ Fácil de entender
- ✅ Menos código
- ✅ Misma funcionalidad core
- ❌ Sin tests
- ❌ Sin evaluación avanzada

## 📄 Licencia

MIT

---

**¿Dudas?** El código en `alpr.py` está bien comentado. Lee ese archivo para entender cómo funciona todo.
