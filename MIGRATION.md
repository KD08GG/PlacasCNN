# Migración a Versión Simplificada

## 🎯 ¿Qué cambió?

El proyecto ahora tiene **2 versiones**:

### ✨ Versión SIMPLE (NUEVA - RECOMENDADA)

**Archivos principales:**
- `alpr.py` - TODO el sistema en UN archivo
- `train.py` - Entrenamiento simplificado
- `README_SIMPLE.md` - Documentación simple
- `requirements_simple.txt` - Dependencias mínimas

**Ventajas:**
- ✅ Solo 2 archivos de código
- ✅ Fácil de entender
- ✅ Menos de 400 líneas total
- ✅ Misma funcionalidad core

**Úsala si:**
- Quieres algo simple y directo
- No necesitas tests ni evaluación avanzada
- Prefieres código en menos archivos

### 📦 Versión COMPLETA (ANTERIOR)

**Estructura modular:**
- `detectors/`, `segmenters/`, `recognizers/`, `pipeline/`
- Multiple archivos y scripts
- Tests unitarios
- Sistema de evaluación
- Notebooks

**Úsala si:**
- Necesitas código muy modular
- Quieres tests y evaluación completa
- Planeas extender mucho el sistema

## 🚀 Cómo usar la Versión Simple

### Opción 1: Solo archivos nuevos

```bash
# Instalar dependencias mínimas
pip install -r requirements_simple.txt

# Generar datos
python train.py synthetic --samples 100

# Entrenar
python train.py classifier --data data/synthetic_chars

# Usar
python alpr.py --image foto.jpg
```

### Opción 2: Empezar de cero

```bash
# Crear carpeta nueva
mkdir PlacasSimple
cd PlacasSimple

# Copiar solo archivos simples
cp ../PlacasCNN/alpr.py .
cp ../PlacasCNN/train.py .
cp ../PlacasCNN/requirements_simple.txt requirements.txt
cp ../PlacasCNN/README_SIMPLE.md README.md

# Listo!
pip install -r requirements.txt
```

## 🔄 Equivalencias

| Versión Completa | Versión Simple |
|------------------|----------------|
| `python predict.py --image foto.jpg` | `python alpr.py --image foto.jpg` |
| `python train_yolo.py --data data.yaml` | `python train.py detector --data data.yaml` |
| `python train_classifier.py --data chars/` | `python train.py classifier --data chars/` |
| `python generate_synthetic_chars.py` | `python train.py synthetic` |
| Pipeline en `pipeline/alpr_pipeline.py` | Todo en `alpr.py` |

## 📚 Documentación

- **Simple**: Lee `README_SIMPLE.md`
- **Completa**: Lee `README.md`

## 💡 Recomendación

**Para la mayoría de usuarios, usa la versión SIMPLE.**

Solo usa la completa si realmente necesitas la modularidad extra.

## 🗑️ Limpiar archivos antiguos (opcional)

Si solo quieres la versión simple:

```bash
# Respaldar versión completa
mkdir backup_completa
mv detectors segmenters recognizers pipeline utils tests notebooks backup_completa/
mv predict.py evaluate.py extract_chars.py generate_synthetic_chars.py backup_completa/
mv setup.py validate_environment.py run_tests.py setup_dataset.py backup_completa/

# Renombrar archivos simples
mv requirements_simple.txt requirements.txt
mv README_SIMPLE.md README.md

# Listo! Solo quedan alpr.py y train.py
```

## ❓ FAQ

**Q: ¿Perdemos funcionalidad con la versión simple?**
A: No. La funcionalidad CORE es la misma. Solo pierdes evaluación avanzada, tests y notebooks.

**Q: ¿Puedo usar ambas versiones?**
A: Sí! Los archivos son independientes.

**Q: ¿Cuál es más rápida?**
A: Ambas tienen el mismo rendimiento. La simple es solo más fácil de leer.

**Q: ¿Y si ya entrené modelos con la versión completa?**
A: Los modelos son compatibles! Usa los mismos paths en `alpr.py`.

---

**En resumen:** La versión SIMPLE es perfecta para la mayoría de casos. Es el mismo sistema, solo más directo.
