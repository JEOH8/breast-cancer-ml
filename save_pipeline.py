# -*- coding: utf-8 -*-
"""
save_pipeline.py
================
Guarda el pipeline completo (scaler + PCA + modelo optimizado) en un
único archivo joblib para ser cargado por la demo de Streamlit.

Uso:
    conda activate breast_cancer
    python save_pipeline.py
"""

import os, sys
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import download_dataset, load_data, preprocess
from models import split_data
from pca_analysis import run_pca
from optimization import bayesian_optimization

print("=" * 60)
print("GUARDANDO PIPELINE COMPLETO PARA STREAMLIT")
print("=" * 60)

# 1. Datos
print("\n[1/4] Cargando datos...")
path = download_dataset()
raw  = load_data(path)
X_imputed, y, _ = preprocess(raw)

# 2. Split
print("\n[2/4] Split train/test...")
X_train_raw, X_test_raw, y_train, y_test = split_data(X_imputed, y)

# 3. Scaler + PCA (fit solo en train)
print("\n[3/4] Ajustando scaler y PCA en train...")
pca_optimal, X_train_pca, X_test_pca, scaler = run_pca(X_train_raw, X_test_raw)

# 4. Modelo optimizado
print("\n[4/4] Entrenando modelo optimizado...")
_, optimized_model = bayesian_optimization(X_train_pca, y_train, n_iter=50)

# Guardar pipeline completo como objeto sklearn Pipeline
full_pipeline = Pipeline([
    ("scaler", scaler),
    ("pca",    pca_optimal),
    ("model",  optimized_model),
])

os.makedirs("models", exist_ok=True)
pipeline_path = "models/full_pipeline.joblib"
joblib.dump(full_pipeline, pipeline_path)

# Guardar también los nombres de las features para la UI
feature_names = list(X_imputed.columns)
joblib.dump(feature_names, "models/feature_names.joblib")

# Guardar estadísticas de las features para los sliders
feature_stats = {
    "min":  X_imputed.min().to_dict(),
    "max":  X_imputed.max().to_dict(),
    "mean": X_imputed.mean().to_dict(),
}
joblib.dump(feature_stats, "models/feature_stats.joblib")

print(f"\n✓ Pipeline guardado en: {pipeline_path}")
print(f"  Features: {len(feature_names)}")
print(f"  Pasos   : {[step[0] for step in full_pipeline.steps]}")

# Verificación rápida
import pandas as pd
sample = X_imputed.iloc[:5]
preds  = full_pipeline.predict(sample)
probas = full_pipeline.predict_proba(sample)[:, 1]
print(f"\nVerificacion (5 muestras):")
for i, (pred, prob) in enumerate(zip(preds, probas)):
    print(f"  Muestra {i+1}: {'Maligno' if pred==1 else 'Benigno'} "
          f"(probabilidad maligno: {prob:.4f})")

print("\n✓ Pipeline listo para Streamlit.")
