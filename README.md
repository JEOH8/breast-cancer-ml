# 🔬 Clasificación de Cáncer de Mama con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-ff6f00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Proyecto de clasificación binaria para diagnóstico de cáncer de mama (benigno/maligno) utilizando el **Wisconsin Breast Cancer Dataset**. El pipeline completo incluye análisis exploratorio, reducción de dimensionalidad con PCA, comparativa de modelos de clasificación, optimización bayesiana de hiperparámetros y una red neuronal profunda.

---

## 📋 Tabla de Contenidos

- [Descripción del Problema](#descripción-del-problema)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline del Proyecto](#pipeline-del-proyecto)
- [Resultados](#resultados)
- [Instalación y Uso](#instalación-y-uso)
- [Tecnologías](#tecnologías)

---

## 🎯 Descripción del Problema

El cáncer de mama es uno de los cánceres más comunes a nivel mundial. La detección temprana y precisa es crítica para mejorar las tasas de supervivencia. Este proyecto desarrolla un sistema de clasificación automática capaz de distinguir tumores **benignos** y **malignos** a partir de características morfológicas celulares, con el objetivo de asistir al personal médico en el proceso diagnóstico.

---

## 📊 Dataset

- **Fuente:** [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer) — Kaggle
- **Muestras:** 569 registros
- **Características:** 30 variables numéricas (radio, textura, perímetro, área, suavidad, etc.)
- **Variable objetivo:** `diagnosis` → `B` (Benigno) / `M` (Maligno)
- **Distribución de clases:** ~63% Benigno / ~37% Maligno

---

## 📁 Estructura del Proyecto

```
breast-cancer-ml/
│
├── 📓 notebooks/
│   └── breast_cancer_analysis.ipynb   # Notebook principal con análisis completo
│
├── 🐍 src/
│   ├── preprocessing.py               # Carga y limpieza de datos
│   ├── eda.py                         # Análisis exploratorio de datos
│   ├── pca_analysis.py                # Análisis de Componentes Principales
│   ├── models.py                      # Entrenamiento y evaluación de modelos
│   ├── optimization.py                # Optimización bayesiana de hiperparámetros
│   └── neural_network.py              # Red neuronal profunda con TensorFlow
│
├── 📊 reports/
│   └── figures/                       # Gráficos y visualizaciones exportadas
│
├── 🤖 models/                         # Modelos entrenados (generados al ejecutar)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline del Proyecto

```
Datos Raw
    │
    ▼
1️⃣  Carga y Exploración Inicial
    │  • Dimensiones, tipos de datos, valores nulos
    │  • Distribución de la variable objetivo
    ▼
2️⃣  Preprocesamiento
    │  • Eliminación de columnas irrelevantes (id, Unnamed: 32)
    │  • Codificación de la variable objetivo (B→0, M→1)
    │  • Imputación de valores nulos (mediana)
    ▼
3️⃣  Análisis Exploratorio (EDA)
    │  • Matriz de correlación
    │  • Boxplots por diagnóstico + test t de Student
    │  • Histogramas + pruebas de normalidad (KS)
    │  • Pairplot con hue por clase
    ▼
4️⃣  Reducción de Dimensionalidad (PCA)
    │  • PCA exploratorio (todos los componentes)
    │  • Selección óptima: 17 componentes → 99% varianza
    │  • Visualización en 2D con elipses de confianza
    │  • Análisis de cargas (loadings)
    ▼
5️⃣  Modelos de Clasificación
    │  • Regresión Logística
    │  • K-Nearest Neighbors (k=5)
    │  • Support Vector Machine (RBF)
    │  • Árbol de Decisión (max_depth=5)
    │  • Random Forest (100 estimadores)
    ▼
6️⃣  Comparativa de Modelos
    │  • Accuracy, Precision, Recall, F1-Score, AUC-ROC
    │  • Curvas ROC comparativas
    │  • Matrices de confusión
    ▼
7️⃣  Optimización Bayesiana (BayesSearchCV)
    │  • Búsqueda en espacio continuo de hiperparámetros
    │  • 50 iteraciones, CV-5 estratificado
    │  • Métrica objetivo: F1-Score
    ▼
8️⃣  Validación Cruzada (10-fold estratificado)
    │  • Estabilidad del modelo optimizado
    │  • Análisis de varianza entre folds
    ▼
9️⃣  Red Neuronal Profunda (TensorFlow/Keras)
    │  • Arquitectura: 64→32→16→1
    │  • Regularización: Dropout + L2 + BatchNorm
    │  • Callbacks: EarlyStopping, ReduceLROnPlateau
    │  • Optimización del umbral de decisión
    ▼
🏆  Sistema Predictivo Final
       • Función de predicción con salida interpretable
       • Guardado de modelos con metadatos
       • Comparativa final: Regresión Logística vs Red Neuronal
```

---

## 📈 Resultados

### Comparativa de Modelos de Clasificación

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| Regresión Logística (base) | 0.982456 | 1.000000 | 0.952381 | 0.975610 | 0.996693 |
| K-Nearest Neighbors (k=5) | 0.956140 | 0.974359 | 0.904762 | 0.938272 | 0.982308 |
| SVM (RBF) | 0.964912 | 0.975000 | 0.928571 | 0.951220 | 0.995040 |
| Árbol de Decisión | 0.947368 | 0.950000 | 0.904762 | 0.926829 | 0.938492 |
| Random Forest | 0.947368 | 0.950000 | 0.904762 | 0.926829 | 0.988757 |
| **Reg. Logística (optimizada)** | 0.991228 | 1.00000 | 0.976190 | 0.987952 | 0.997685 |
| **Red Neuronal Profunda** | 0.991228 | 1.000000 | 0.976190 | 0.987952 | 0.999339 |


### Reducción de Dimensionalidad

- **Dimensionalidad original:** 30 características
- **Componentes PCA seleccionados:** 17
- **Varianza explicada:** 99%
- **Tasa de compresión:** 43%

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/breast-cancer-ml.git
cd breast-cancer-ml
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configurar credenciales de Kaggle

Para descargar el dataset automáticamente necesitas una cuenta en [Kaggle](https://www.kaggle.com/) y tu API key (`kaggle.json`):

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Ejecutar el notebook

```bash
jupyter notebook notebooks/breast_cancer_analysis.ipynb
```

O ejecutar cada módulo de forma independiente:

```bash
python src/preprocessing.py
python src/eda.py
python src/pca_analysis.py
python src/models.py
python src/optimization.py
python src/neural_network.py
```

---

## 🛠️ Tecnologías

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.10+ |
| Manipulación de datos | NumPy, Pandas |
| Visualización | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Optimización | Scikit-Optimize (BayesSearchCV) |
| Deep Learning | TensorFlow / Keras |
| Dataset | KaggleHub |
| Entorno | Jupyter Notebook |

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">Desarrollado como parte de mi portafolio de Data Science & Machine Learning</p>
