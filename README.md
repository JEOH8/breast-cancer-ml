# 🔬 Clasificación de Cáncer de Mama con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-ff6f00?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b?logo=streamlit)](https://breast-cancer-ml-jeoh1.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de clasificación binaria para diagnóstico asistido de cáncer de mama (benigno/maligno) sobre el **Wisconsin Breast Cancer Dataset**. El pipeline incluye EDA, PCA, comparativa de 5 clasificadores, optimización bayesiana de hiperparámetros, red neuronal profunda y verificación explícita de data leakage.

> Este sistema puede asistir al personal médico en la priorización de casos, reduciendo el tiempo de diagnóstico en contextos con recursos limitados.

🔗 **[Demo interactiva en Streamlit](https://breast-cancer-ml-jeoh1.streamlit.app/)** — predicción manual por sliders y predicción por lote (CSV).

---

## 📋 Tabla de Contenidos

- [Contexto Clínico](#contexto-clínico)
- [Dataset](#dataset)
- [Demo Interactiva](#demo-interactiva)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline del Proyecto](#pipeline-del-proyecto)
- [Resultados](#resultados)
- [Verificación de Data Leakage](#verificación-de-data-leakage)
- [Instalación y Uso](#instalación-y-uso)
- [Tecnologías](#tecnologías)

---

## 🏥 Contexto Clínico

Las 30 características morfológicas de este modelo **no provienen de mamografías**. Se calculan computacionalmente a partir de imágenes digitalizadas de **biopsias de aspiración con aguja fina (FNA)**, analizadas bajo microscopio. Variables como radio, textura, concavidad y dimensión fractal son extraídas automáticamente por software especializado de análisis de imagen (originalmente WBC — Wisconsin Breast Cytology).

En un entorno clínico real, el flujo sería:

```
Imagen de biopsia FNA digitalizada
        ↓
Software de segmentación celular (ej. ImageJ + WBC plugin)
        ↓
Extracción automática de las 30 características morfológicas
        ↓
Modelo predictivo → Diagnóstico asistido
```

⚠️ **Este sistema es una herramienta académica e investigativa. No reemplaza el diagnóstico clínico profesional.**

---

## 📊 Dataset

- **Fuente:** [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer) — Kaggle
- **Muestras:** 569 registros
- **Características:** 30 variables numéricas (radio, textura, perímetro, área, concavidad, etc.)
- **Variable objetivo:** `diagnosis` → `B` (Benigno) / `M` (Maligno)
- **Distribución de clases:** ~63% Benigno / ~37% Maligno

---

## 🚀 Demo Interactiva

**[→ Abrir demo en Streamlit](https://breast-cancer-ml.streamlit.app)**

La demo ofrece dos modos de uso:

| Modo | Descripción |
|------|-------------|
| 🧪 Predicción Manual | Ajuste de las 30 características morfológicas mediante sliders interactivos |
| 📂 Predicción por Lote | Subida de CSV con múltiples pacientes — incluye plantilla descargable |

Incluye también una pestaña con métricas completas del modelo, tabla comparativa de clasificadores y verificación de data leakage.

Para ejecutar la demo localmente:

```bash
python save_pipeline.py   # genera el pipeline serializado
streamlit run app.py
```

---

## 📁 Estructura del Proyecto

```
breast-cancer-ml/
│
├── 📓 notebooks/
│   └── breast_cancer_analysis.ipynb   # Notebook principal reproducible
│
├── 🐍 src/
│   ├── preprocessing.py               # Carga y limpieza estructural
│   ├── eda.py                         # Análisis exploratorio de datos
│   ├── pca_analysis.py                # PCA sin data leakage
│   ├── models.py                      # Entrenamiento, CV y comparativa
│   ├── optimization.py                # Optimización Bayesiana + verificación leakage
│   └── neural_network.py              # Red neuronal + verificación leakage
│
├── 📈 reports/figures/                # Gráficos exportados
├── 🗂️ models/                         # Pipeline y modelos serializados
├── app.py                             # Demo interactiva (Streamlit)
├── save_pipeline.py                   # Genera pipeline para Streamlit
├── main.py                            # Pipeline completo desde terminal
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline del Proyecto

```
Datos Raw
    │
    ▼
1️⃣  Carga y Exploración Inicial
    │  • Dimensiones, tipos, valores nulos
    │  • Distribución de la variable objetivo
    ▼
2️⃣  Preprocesamiento Estructural
    │  • Eliminación de columnas irrelevantes
    │  • Codificación: B→0, M→1
    ▼
3️⃣  Análisis Exploratorio (EDA)
    │  • Matriz de correlación (21 pares con |r| > 0.9)
    │  • Boxplots por diagnóstico + test t de Student
    │  • Histogramas + pruebas de normalidad (KS)
    ▼
4️⃣  Split estratificado train/test ← ANTES de cualquier transformación
    │  • 80% entrenamiento / 20% prueba
    │  • Estratificado por clase
    ▼
5️⃣  Reducción de Dimensionalidad (PCA sin data leakage)
    │  • StandardScaler: fit SOLO en train
    │  • PCA: fit SOLO en train → 17 componentes → 99.1% varianza
    │  • Tasa de compresión: 43%
    ▼
6️⃣  Validación Cruzada — 5 modelos base (10-fold estratificado)
    │  • Regresión Logística, KNN, SVM, Árbol de Decisión, Random Forest
    │  • Comparativa CV vs holdout (verificación de leakage)
    ▼
7️⃣  Optimización Bayesiana (BayesSearchCV)
    │  • 50 iteraciones, CV-5, métrica F1
    │  • Espacio continuo de hiperparámetros
    ▼
8️⃣  Validación Cruzada del Modelo Optimizado (10-fold)
    │  • Estabilidad y varianza entre folds
    ▼
9️⃣  Red Neuronal Profunda (TensorFlow/Keras)
    │  • Arquitectura: 64→32→16→1
    │  • Dropout + L2 + BatchNormalization
    │  • EarlyStopping, ReduceLROnPlateau
    │  • Optimización del umbral de decisión
    ▼
🎯  Verificación explícita de data leakage
       • CV vs holdout para Reg. Logística optimizada
       • Train vs Val vs Holdout para Red Neuronal
```

---

## 📊 Resultados

### Comparativa de Modelos de Clasificación

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| Regresión Logística (base) | 0.982456 | 1.000000 | 0.952381 | 0.975610 | 0.996693 |
| K-Nearest Neighbors (k=5) | 0.956140 | 0.974359 | 0.904762 | 0.938272 | 0.982308 |
| SVM (RBF) | 0.964912 | 0.975000 | 0.928571 | 0.951220 | 0.995370 |
| Árbol de Decisión | 0.947368 | 0.950000 | 0.904762 | 0.926829 | 0.938161 |
| Random Forest | 0.947368 | 0.950000 | 0.904762 | 0.926829 | 0.991898 |
| **Reg. Logística (optimizada)** ⭐ | **0.991228** | **1.000000** | **0.976190** | **0.987952** | **0.998016** |
| **Red Neuronal Profunda** | **0.991228** | **1.000000** | **0.976190** | **0.987952** | 0.995–0.999* |

*AUC de la red neuronal varía entre ejecuciones por aleatoriedad de TensorFlow en CPU — comportamiento documentado.

### Reducción de Dimensionalidad

| Métrica | Valor |
|---------|-------|
| Dimensionalidad original | 30 características |
| Componentes PCA seleccionados | 17 |
| Varianza explicada | 99.1% |
| Tasa de compresión | 43% |
| Pares con correlación \|r\| > 0.9 eliminados | 21 |

### Conclusiones Clave

1. **PCA fue determinante:** eliminó 21 pares de variables altamente correlacionadas, mejorando la generalización de todos los modelos.
2. **Precision = 1.0 en todas las ejecuciones** para el modelo recomendado: cuando dice "Maligno", siempre tiene razón. Cero falsos positivos.
3. **Regresión Logística optimizada ≈ Red Neuronal** en todas las métricas críticas, con mayor estabilidad y reproducibilidad. En contextos clínicos, la interpretabilidad y estabilidad importan tanto como el rendimiento pico.
4. **Optimización Bayesiana** superó a Grid Search encontrando la configuración óptima en 50 iteraciones.

---

## ✅ Verificación de Data Leakage

El pipeline fue auditado y corregido tras feedback externo de un PhD researcher en ML biomédico.

**Corrección aplicada:** El `StandardScaler` y el `PCA` ahora se ajustan **exclusivamente sobre el conjunto de entrenamiento**, eliminando cualquier contaminación de información del test set.

### Regresión Logística Optimizada (CV-10 vs Holdout)

| Métrica | CV-10 (train) | Holdout (test) | Δ | Estado |
|---------|--------------|----------------|---|--------|
| Accuracy | 97.58% | 99.12% | +1.54% | ✅ |
| Precision | 97.18% | 100.00% | +2.82% | ⚠️ monitorear* |
| Recall | 96.47% | 97.62% | +1.15% | ✅ |
| F1 | 96.73% | 98.80% | +2.06% | ⚠️ monitorear* |
| AUC | 99.42% | 99.80% | +0.38% | ✅ |

*Diferencia positiva (holdout > CV) no indica leakage — el leakage se manifiesta como caída en test. Varianza esperable con 114 muestras de prueba.

### Red Neuronal Profunda (Train vs Val vs Holdout)

| Métrica | Train | Val (mejor época) | Holdout | Δ Val→Hold | Estado |
|---------|-------|-------------------|---------|------------|--------|
| Accuracy | 0.9736 | 0.9825 | 0.9912 | +0.0088 | ✅ |
| Precision | 0.9702 | 1.0000 | 1.0000 | +0.0000 | ✅ |
| Recall | 0.9588 | 0.9524 | 0.9762 | +0.0238 | ✅ |
| AUC | 0.9939 | 0.9979 | 0.9977 | -0.0002 | ✅ |

---

## 🛠️ Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/JEOH8/breast-cancer-ml.git
cd breast-cancer-ml
```

### 2. Crear entorno e instalar dependencias

```bash
conda create -n breast_cancer python=3.9
conda activate breast_cancer
pip install -r requirements.txt
```

### 3. Configurar credenciales de Kaggle

```bash
# Descargar kaggle.json desde kaggle.com → Settings → API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json   # Linux/Mac
```

### 4. Ejecutar el pipeline completo

```bash
python main.py              # Pipeline completo
python main.py --skip-eda   # Sin gráficos exploratorios
python main.py --skip-nn    # Sin red neuronal
```

### 5. Ejecutar la demo de Streamlit localmente

```bash
python save_pipeline.py     # Genera el pipeline serializado (solo primera vez)
streamlit run app.py
```

### 6. Ejecutar módulos individuales

```bash
python src/preprocessing.py
python src/eda.py
python src/pca_analysis.py
python src/models.py
python src/optimization.py
python src/neural_network.py
```

---

## 🔧 Tecnologías

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.9+ |
| Manipulación de datos | NumPy, Pandas |
| Visualización | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-Learn |
| Optimización | Scikit-Optimize (BayesSearchCV) |
| Deep Learning | TensorFlow / Keras |
| Demo interactiva | Streamlit |
| Dataset | KaggleHub |
| Serialización | Joblib |

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
Desarrollado por <strong>Juan Esteban Ospina</strong> — Ingeniero Biomédico<br>
<a href="https://github.com/JEOH8">github.com/JEOH8</a>
</p>
