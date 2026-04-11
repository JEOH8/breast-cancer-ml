# -*- coding: utf-8 -*-
"""
main.py
=======
Pipeline completo de clasificación de cáncer de mama.
Corregido para eliminar data leakage.

ORDEN CORRECTO DEL PIPELINE:
    1. Carga y preprocesamiento estructural
    2. Split train/test  ← ANTES de cualquier transformación
    3. PCA (fit solo en train, transform en ambos)
    4. Validación cruzada de los 5 modelos base
    5. Entrenamiento y evaluación en holdout test
    6. Comparativa CV vs holdout (verificación de leakage)
    7. Optimización Bayesiana
    8. Validación cruzada del modelo optimizado
    9. Red Neuronal Profunda

Uso:
    conda activate breast_cancer
    python main.py
    python main.py --skip-eda
    python main.py --skip-nn
    python main.py --skip-eda --skip-nn
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import download_dataset, load_data, preprocess
from eda import run_eda
from pca_analysis import run_pca, plot_2d_with_labels
from models import (split_data, train_all_models, plot_comparison,
                    cross_validate_all_models, plot_cv_vs_holdout)
from optimization import bayesian_optimization, cross_validate, compare_models
from neural_network import train_neural_network, save_model, verify_no_leakage_nn
from ablation_study import run_ablation_study, plot_ablation_results, print_summary_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de clasificación de cáncer de mama (sin data leakage)"
    )
    parser.add_argument("--skip-eda", action="store_true",
                        help="Omite el análisis exploratorio")
    parser.add_argument("--skip-nn",  action="store_true",
                        help="Omite la red neuronal")
    parser.add_argument("--ablation", action="store_true",
                        help="Ejecuta ablation study de arquitecturas de red neuronal")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  PIPELINE: CLASIFICACIÓN DE CÁNCER DE MAMA")
    print("  Versión corregida — sin data leakage")
    print("=" * 70)

    # ── 1. Datos ──────────────────────────────────────────────────────────
    print("\n[1/9] Cargando datos...")
    file_path = download_dataset()
    raw_data  = load_data(file_path)

    # ── 2. Preprocesamiento estructural ───────────────────────────────────
    print("\n[2/9] Preprocesamiento estructural...")
    X_imputed, y, data_processed = preprocess(raw_data)

    # ── 3. EDA (sobre datos completos, antes del split, solo exploratorio)
    if not args.skip_eda:
        print("\n[3/9] Análisis exploratorio de datos...")
        run_eda(X_imputed, y, data_processed)
    else:
        print("\n[3/9] EDA omitido (--skip-eda)")

    # ── 4. Split PRIMERO — antes de cualquier transformación ─────────────
    print("\n[4/9] División train/test (ANTES del escalado y PCA)...")
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_imputed, y)

    # ── 5. PCA (fit solo en train) ────────────────────────────────────────
    print("\n[5/9] PCA sin data leakage...")
    pca_optimal, X_train_pca, X_test_pca, scaler = run_pca(X_train_raw, X_test_raw)
    plot_2d_with_labels(X_train_pca, y_train, pca_optimal,
                        "PCA — Conjunto de Entrenamiento")

    # ── 6. CV de los 5 modelos base + holdout test ────────────────────────
    print("\n[6/9] Validación cruzada (5 modelos) + evaluación holdout...")
    cv_summary = cross_validate_all_models(X_train_pca, y_train)
    resultados, models_dict = train_all_models(
        X_train_pca, X_test_pca, y_train, y_test
    )
    plot_cv_vs_holdout(cv_summary, resultados, metric="F1")
    plot_comparison(resultados, models_dict, X_test_pca, y_test, pca_optimal)

    # ── 7. Optimización Bayesiana ─────────────────────────────────────────
    print("\n[7/9] Optimización Bayesiana...")
    from sklearn.linear_model import LogisticRegression
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_train_pca, y_train)

    bayes_search, optimized_model = bayesian_optimization(X_train_pca, y_train)

    # ── 8. Validación cruzada del modelo optimizado ───────────────────────
    print("\n[8/9] Validación cruzada del modelo optimizado...")
    cv_results = cross_validate(optimized_model, X_train_pca, y_train)
    base_metrics, opt_metrics = compare_models(
        base_model, optimized_model,
        X_train_pca, X_test_pca, y_train, y_test
    )

    # Verificación de leakage — Reg. Logística optimizada
    from optimization import verify_no_leakage_optimized
    verify_no_leakage_optimized(cv_results, opt_metrics, base_metrics)

    # ── 9. Red Neuronal ───────────────────────────────────────────────────
    if not args.skip_nn:
        print("\n[9/9] Red Neuronal Profunda...")
        os.makedirs("models", exist_ok=True)
        best_nn, history, best_threshold = train_neural_network(
            X_train_pca, X_test_pca, y_train, y_test
        )

        from sklearn.metrics import (
            accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score
        )
        test_preds    = best_nn.predict(X_test_pca.values).flatten()
        test_pred_opt = (test_preds > best_threshold).astype(int)

        print("\nMÉTRICAS FINALES — Red Neuronal Profunda (umbral óptimo)")
        print(f"  Accuracy  : {accuracy_score(y_test, test_pred_opt):.6f}")
        print(f"  Precision : {precision_score(y_test, test_pred_opt):.6f}")
        print(f"  Recall    : {recall_score(y_test, test_pred_opt):.6f}")
        print(f"  F1-Score  : {f1_score(y_test, test_pred_opt):.6f}")
        print(f"  AUC       : {roc_auc_score(y_test, test_preds):.6f}")

        # Verificación de leakage — Red Neuronal
        verify_no_leakage_nn(best_nn, history,
                              X_train_pca, X_test_pca,
                              y_train, y_test, best_threshold)

        save_model(best_nn, history, best_threshold,
                   X_train_pca, pca_optimal, opt_metrics)
    else:
        print("\n[9/9] Red neuronal omitida (--skip-nn)")

    # ── Ablation study (opcional) ────────────────────────────────────────
    if args.ablation:
        print("\n[Ablation] Ejecutando ablation study de arquitecturas...")
        results_df, histories = run_ablation_study(
            X_train_pca, X_test_pca, y_train, y_test
        )
        plot_ablation_results(results_df, histories)
        print_summary_table(results_df)

    # ── Resumen final ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETADO — VERSIÓN SIN DATA LEAKAGE")
    print("=" * 70)
    print(f"\n  Mejor modelo clásico : Reg. Logística Optimizada")
    print(f"    Accuracy  : {opt_metrics['Accuracy Test']:.6f}")
    print(f"    Precision : {opt_metrics['Precision Test']:.6f}")
    print(f"    Recall    : {opt_metrics['Recall Test']:.6f}")
    print(f"    F1-Score  : {opt_metrics['F1 Test']:.6f}")
    print(f"    AUC       : {opt_metrics['AUC Test']:.6f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
