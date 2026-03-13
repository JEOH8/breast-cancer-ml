# -*- coding: utf-8 -*-
"""
main.py
=======
Pipeline completo de clasificación de cáncer de mama.

Uso:
    conda activate breast_cancer
    cd C:\\Users\\Juan Esteban\\breastcancergithub\\breast-cancer-ml
    python main.py
    python main.py --skip-eda        # omite gráficos exploratorios
    python main.py --skip-nn         # omite red neuronal (más rápido)

Pipeline:
    1. Descarga y carga del dataset (Kaggle)
    2. Preprocesamiento
    3. EDA
    4. PCA
    5. Modelos de clasificación (LR, KNN, SVM, DT, RF)
    6. Optimización Bayesiana
    7. Validación cruzada
    8. Red Neuronal Profunda
    9. Guardado de modelos y resultados
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Añadir src/ al path para que los imports funcionen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import download_dataset, load_data, preprocess
from eda import run_eda
from pca_analysis import run_pca, plot_2d_with_labels
from models import split_data, train_all_models, plot_comparison
from optimization import bayesian_optimization, cross_validate, compare_models
from neural_network import train_neural_network, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de clasificación de cáncer de mama")
    parser.add_argument("--skip-eda", action="store_true", help="Omite el análisis exploratorio")
    parser.add_argument("--skip-nn",  action="store_true", help="Omite la red neuronal")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  PIPELINE: CLASIFICACIÓN DE CÁNCER DE MAMA CON MACHINE LEARNING")
    print("=" * 70)

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    print("\n[1/8] Cargando datos...")
    file_path = download_dataset()
    raw_data  = load_data(file_path)

    # ── 2. Preprocesamiento ───────────────────────────────────────────────────
    print("\n[2/8] Preprocesamiento...")
    X_imputed, y, data_processed = preprocess(raw_data)

    # ── 3. EDA ────────────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("\n[3/8] Análisis exploratorio de datos...")
        run_eda(X_imputed, y, data_processed)
    else:
        print("\n[3/8] EDA omitido (--skip-eda)")

    # ── 4. PCA ────────────────────────────────────────────────────────────────
    print("\n[4/8] PCA...")
    pca_optimal, X_pca_df, scaler = run_pca(X_imputed)
    plot_2d_with_labels(X_pca_df, y, pca_optimal)

    # ── 5. Modelos clásicos ───────────────────────────────────────────────────
    print("\n[5/8] Entrenando modelos de clasificación...")
    X_train, X_test, y_train, y_test = split_data(X_pca_df, y)
    resultados, models_dict = train_all_models(X_train, X_test, y_train, y_test)
    plot_comparison(resultados, models_dict, X_test, y_test, pca_optimal)

    # ── 6. Optimización Bayesiana ─────────────────────────────────────────────
    print("\n[6/8] Optimización Bayesiana...")
    from sklearn.linear_model import LogisticRegression
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_train, y_train)

    bayes_search, optimized_model = bayesian_optimization(X_train, y_train)

    # ── 7. Validación cruzada ─────────────────────────────────────────────────
    print("\n[7/8] Validación cruzada...")
    cv_results = cross_validate(optimized_model, X_train, y_train)
    base_metrics, opt_metrics = compare_models(
        base_model, optimized_model, X_train, X_test, y_train, y_test
    )

    # ── 8. Red Neuronal ───────────────────────────────────────────────────────
    if not args.skip_nn:
        print("\n[8/8] Red Neuronal Profunda...")
        os.makedirs("models", exist_ok=True)
        best_nn, history, best_threshold = train_neural_network(
            X_train, X_test, y_train, y_test
        )

        # Métricas finales reproducibles
        from sklearn.metrics import (
            accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score
        )
        import tensorflow as tf
        test_preds = best_nn.predict(
            X_test.values if hasattr(X_test, "values") else X_test
        ).flatten()
        test_pred_opt = (test_preds > best_threshold).astype(int)

        print("\nMÉTRICAS FINALES — Red Neuronal Profunda (umbral óptimo)")
        print(f"  Accuracy  : {accuracy_score(y_test, test_pred_opt):.6f}")
        print(f"  Precision : {precision_score(y_test, test_pred_opt):.6f}")
        print(f"  Recall    : {recall_score(y_test, test_pred_opt):.6f}")
        print(f"  F1-Score  : {f1_score(y_test, test_pred_opt):.6f}")
        print(f"  AUC       : {roc_auc_score(y_test, test_preds):.6f}")

        save_model(best_nn, history, best_threshold, X_train, pca_optimal, opt_metrics)
    else:
        print("\n[8/8] Red neuronal omitida (--skip-nn)")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\n  Modelos guardados en: models/")
    print(f"  Mejor modelo clásico : Reg. Logística Optimizada")
    print(f"    Accuracy  : {opt_metrics['Accuracy Test']:.6f}")
    print(f"    Precision : {opt_metrics['Precision Test']:.6f}")
    print(f"    Recall    : {opt_metrics['Recall Test']:.6f}")
    print(f"    F1-Score  : {opt_metrics['F1 Test']:.6f}")
    print(f"    AUC       : {opt_metrics['AUC Test']:.6f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()