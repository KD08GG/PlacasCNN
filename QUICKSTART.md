# Guía de Inicio Rápido

Esta guía te permitirá empezar a usar el sistema ALPR en menos de 10 minutos.

## ⚡ Inicio Rápido (5 minutos)

### 1. Clonar e Instalar

```bash
# Clonar repositorio
git clone <url-del-repo>
cd PlacasCNN

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar setup
python setup.py
```

### 2. Validar Instalación

```bash
python validate_environment.py
```

Esto verificará que todo esté correctamente instalado.

### 3. Probar con Dataset Sintético

```bash
# Generar dataset sintético de caracteres (para pruebas)
python generate_synthetic_chars.py --output data/char_dataset --samples 50

# Entrenar clasificador (rápido, solo para prueba)
python train_classifier.py --data data/char_dataset --epochs 10

# Hacer predicción (necesitas una imagen de prueba)
python predict.py --image examples/tu_imagen.jpg
```

## 📊 Flujo Completo (con dataset real)

### Paso 1: Preparar Dataset de Detección

```bash
# Descargar dataset de Roboflow (formato YOLOv8)
# Luego descomprimir:
python -c "from setup_dataset import unzip_dataset; unzip_dataset('path/to/dataset.zip')"
```

### Paso 2: Entrenar Detector YOLOv8

```bash
python train_yolo.py --data data/dataset/data.yaml
```

Esto puede tomar varias horas dependiendo de tu hardware.

### Paso 3: Preparar Dataset de Caracteres

**Opción A: Extraer de placas detectadas**

```bash
# Extrae caracteres automáticamente
python extract_chars.py --images carpeta_con_imagenes --output data/char_dataset_extracted

# Organizar interactivamente
python extract_chars.py --organize --images data/char_dataset_extracted/unlabeled --output data/char_dataset
```

**Opción B: Dataset sintético (más rápido)**

```bash
python generate_synthetic_chars.py --output data/char_dataset --samples 200
```

### Paso 4: Entrenar Clasificador

```bash
python train_classifier.py --data data/char_dataset --epochs 50
```

### Paso 5: Predecir

```bash
python predict.py --image examples/test.jpg
```

### Paso 6: Evaluar (opcional)

```bash
# Crear archivo ground_truth.json con tus datos
python evaluate.py --images test_images/ --ground-truth ground_truth.json
```

## 🎓 Usando el Notebook de Ejemplo

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir notebooks/ejemplo_completo.ipynb
```

Sigue las celdas para ver ejemplos interactivos.

## 🔧 Comandos Útiles

### Ejecutar Tests

```bash
python run_tests.py
```

### Generar Dataset Sintético

```bash
python generate_synthetic_chars.py --samples 100
```

### Extraer Caracteres de Imágenes

```bash
python extract_chars.py --images carpeta/ --output data/extracted
```

### Validar Entorno

```bash
python validate_environment.py
```

## 🎯 Casos de Uso Comunes

### Solo Quiero Probar el Sistema

```bash
# 1. Generar datos sintéticos
python generate_synthetic_chars.py --samples 50

# 2. Entrenar rápido (10 epochs)
python train_classifier.py --data data/char_dataset_synthetic --epochs 10

# 3. Predecir (usa modelo preentrenado de YOLO)
python predict.py --image tu_imagen.jpg
```

### Quiero Entrenar con Mis Datos

1. Prepara tu dataset en formato Roboflow
2. Ejecuta `train_yolo.py`
3. Extrae caracteres con `extract_chars.py`
4. Entrena clasificador con `train_classifier.py`
5. Predice con `predict.py`

### Quiero Evaluar el Sistema

```bash
# Crea ground_truth.json con formato:
# {"imagen1.jpg": "ABC123D", "imagen2.jpg": "XYZ789E"}

python evaluate.py --images test_set/ --ground-truth ground_truth.json
```

## ⚠️ Solución Rápida de Problemas

### Error: "No module named 'ultralytics'"

```bash
pip install -r requirements.txt
```

### Error: "No se encontró data.yaml"

El dataset no está descomprimido o no existe. Ejecuta:

```bash
python -c "from setup_dataset import unzip_dataset; unzip_dataset('tu_dataset.zip')"
```

### Error: "No se pudo cargar el modelo clasificador"

No has entrenado el clasificador. Ejecuta:

```bash
python train_classifier.py --data data/char_dataset
```

### El sistema es muy lento

- Asegúrate de tener GPU disponible
- Reduce el tamaño de batch en `config.py`
- Usa modelos más pequeños (yolov8n en lugar de yolov8x)

## 📚 Más Información

- **README.md**: Documentación completa
- **notebooks/ejemplo_completo.ipynb**: Tutorial interactivo
- **examples/README.md**: Ejemplos de uso
- **tests/**: Tests unitarios

## 🆘 Ayuda

Si encuentras problemas:

1. Ejecuta `python validate_environment.py`
2. Revisa los logs en consola
3. Consulta la sección de "Solución de Problemas" en README.md
4. Abre un issue en GitHub

## 🚀 ¡Listo!

Ya estás preparado para usar el sistema ALPR. Comienza con el flujo rápido y luego avanza al flujo completo según tus necesidades.
