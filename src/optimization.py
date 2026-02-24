"""
optimization.py
===============
Optimización Bayesiana de Regresión Logística con BayesSearchCV,
validación cruzada estratificada y comparativa base vs optimizado.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix,
)

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize", "-q"])
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer


# ── Optimización Bayesiana ─────────────────────────────────────────────────

def bayesian_optimization(X_train, y_train, n_iter: int = 50, cv: int = 5):
    """
    Ejecuta BayesSearchCV sobre LogisticRegression.

    Returns
    -------
    bayes_search : BayesSearchCV ajustado
    optimized_model : LogisticRegression
        Mejor estimador reentrenado con todos los datos de entrenamiento.
    """
    print("=" * 70)
    print("OPTIMIZACIÓN BAYESIANA DE REGRESIÓN LOGÍSTICA")
    print("=" * 70)

    search_spaces = {
        "C": Real(0.001, 100, prior="log-uniform"),
        "penalty": Categorical(["l1", "l2"]),
        "solver": Categorical(["liblinear", "saga"]),
        "max_iter": Integer(100, 2000),
        "class_weight": Categorical(["balanced", None]),
        "fit_intercept": Categorical([True, False]),
    }

    print("\n  Espacios de búsqueda:")
    for k, v in search_spaces.items():
        print(f"    • {k}: {v}")

    bayes_search = BayesSearchCV(
        estimator=LogisticRegression(random_state=42),
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=1,
        return_train_score=True,
    )

    print(f"\n  Iteraciones: {n_iter} | CV: {cv}-fold | Métrica: F1-Score")
    print("  Ejecutando búsqueda bayesiana...")

    t0 = time.time()
    bayes_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  ✓ Completado en {elapsed:.1f}s")
    print(f"  Mejor F1 (CV): {bayes_search.best_score_:.4f}")
    print("  Mejores hiperparámetros:")
    for k, v in bayes_search.best_params_.items():
        print(f"    • {k}: {v}")

    optimized_model = bayes_search.best_estimator_
    optimized_model.fit(X_train, y_train)

    return bayes_search, optimized_model


# ── Validación cruzada del modelo optimizado ────────────────────────────────

def cross_validate(model, X_train, y_train, n_splits: int = 10):
    """
    Validación cruzada estratificada para múltiples métricas.

    Returns
    -------
    cv_results : dict[str, np.ndarray]
    """
    print("\n" + "=" * 70)
    print("VALIDACIÓN CRUZADA DEL MODELO OPTIMIZADO")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring_metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = {}
    for name, scoring in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring=scoring, n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name.upper():12s}: {scores.mean():.3%} ± {scores.std():.3%}"
              f"  [{scores.min():.3%} – {scores.max():.3%}]")

    _plot_cv_results(cv_results, n_splits)
    return cv_results


# ── Comparativa base vs optimizado ─────────────────────────────────────────

def compare_models(base_model, optimized_model, X_train, X_test, y_train, y_test):
    """
    Genera métricas y visualizaciones comparando el modelo base con el optimizado.

    Returns
    -------
    base_metrics, opt_metrics : dict, dict
    """
    def compute_metrics(model, X_tr, X_te, y_tr, y_te):
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)
        y_te_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_te_pred
        return {
            "Accuracy Train": accuracy_score(y_tr, y_tr_pred),
            "Accuracy Test": accuracy_score(y_te, y_te_pred),
            "Precision Test": precision_score(y_te, y_te_pred),
            "Recall Test": recall_score(y_te, y_te_pred),
            "F1 Test": f1_score(y_te, y_te_pred),
            "AUC Test": roc_auc_score(y_te, y_te_proba),
        }, y_te_pred, y_te_proba

    base_metrics, base_pred, base_proba = compute_metrics(
        base_model, X_train, X_test, y_train, y_test)
    opt_metrics, opt_pred, opt_proba = compute_metrics(
        optimized_model, X_train, X_test, y_train, y_test)

    # Tabla comparativa
    rows = []
    for metric in base_metrics:
        base_val = base_metrics[metric]
        opt_val = opt_metrics[metric]
        rows.append({
            "Métrica": metric,
            "Modelo Base": f"{base_val:.3%}",
            "Modelo Optimizado": f"{opt_val:.3%}",
            "Mejora Absoluta": f"{opt_val - base_val:+.3%}",
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    _plot_comparison(
        base_metrics, opt_metrics,
        confusion_matrix(y_test, base_pred),
        confusion_matrix(y_test, opt_pred),
        base_proba, opt_proba, y_test, optimized_model,
    )

    return base_metrics, opt_metrics


# ── Funciones de predicción ─────────────────────────────────────────────────

def predict_breast_cancer(features, model, threshold: float = 0.5) -> dict:
    """
    Predicción interpretable para una sola muestra (en espacio PCA).

    Parameters
    ----------
    features : array-like
        Vector de 17 componentes PCA.
    model : estimador ajustado
    threshold : float

    Returns
    -------
    dict con predicción, probabilidades, confianza y recomendación.
    """
    features = np.array(features).reshape(1, -1)
    expected = model.n_features_in_
    if features.shape[1] != expected:
        raise ValueError(f"Se esperaban {expected} características, recibidas {features.shape[1]}.")

    proba = model.predict_proba(features)[0]
    label = "Maligno" if proba[1] >= threshold else "Benigno"
    confidence = proba[1] if label == "Maligno" else proba[0]

    return {
        "prediccion": label,
        "probabilidad_maligno": float(proba[1]),
        "probabilidad_benigno": float(proba[0]),
        "confianza": float(confidence),
        "umbral_utilizado": threshold,
        "recomendacion": (
            "CONSULTAR CON ESPECIALISTA"
            if label == "Maligno"
            else "SEGUIMIENTO RUTINARIO"
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _plot_cv_results(cv_results: dict, n_splits: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cv_data = [{"Métrica": m, "Score": s} for m, scores in cv_results.items() for s in scores]
    cv_df = pd.DataFrame(cv_data)

    sns.boxplot(x="Métrica", y="Score", data=cv_df, ax=axes[0])
    sns.stripplot(x="Métrica", y="Score", data=cv_df, color="black", alpha=0.5, ax=axes[0])
    axes[0].set_title("Distribución de Métricas en Validación Cruzada")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")

    folds = np.arange(1, n_splits + 1)
    for metric in ["accuracy", "f1", "roc_auc"]:
        if metric in cv_results:
            axes[1].plot(folds, cv_results[metric], marker="o", label=metric, linewidth=2)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Puntuación")
    axes[1].set_title("Evolución de Métricas por Fold")
    axes[1].set_xticks(folds)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_comparison(base_m, opt_m, cm_base, cm_opt,
                     base_proba, opt_proba, y_test, opt_model) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    width = 0.35

    # Accuracy
    for i, (label, metrics, color) in enumerate(zip(
        ["Base", "Optimizado"], [base_m, opt_m], ["skyblue", "lightcoral"]
    )):
        vals = [metrics["Accuracy Train"], metrics["Accuracy Test"]]
        axes[0, 0].bar(np.arange(2) + i * width, vals, width, label=label, alpha=0.8, color=color)
    axes[0, 0].set_xticks(np.arange(2) + width / 2)
    axes[0, 0].set_xticklabels(["Entrenamiento", "Prueba"])
    axes[0, 0].set_title("Comparación de Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Precision / Recall / F1
    keys = ["Precision Test", "Recall Test", "F1 Test"]
    for i, (label, metrics, color) in enumerate(zip(
        ["Base", "Optimizado"], [base_m, opt_m], ["skyblue", "lightcoral"]
    )):
        vals = [metrics[k] for k in keys]
        axes[0, 1].bar(np.arange(3) + i * width, vals, width, label=label, alpha=0.8, color=color)
    axes[0, 1].set_xticks(np.arange(3) + width / 2)
    axes[0, 1].set_xticklabels(["Precision", "Recall", "F1"])
    axes[0, 1].set_title("Métricas de Clasificación (Test)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Matriz confusión — base
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[0, 2])
    axes[0, 2].set_title("Confusión — Modelo Base")
    axes[0, 2].set_xticklabels(["Benigno", "Maligno"])
    axes[0, 2].set_yticklabels(["Benigno", "Maligno"])

    # Matriz confusión — optimizado
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", ax=axes[1, 0])
    axes[1, 0].set_title("Confusión — Modelo Optimizado")
    axes[1, 0].set_xticklabels(["Benigno", "Maligno"])
    axes[1, 0].set_yticklabels(["Benigno", "Maligno"])

    # Curvas ROC
    fpr_b, tpr_b, _ = roc_curve(y_test, base_proba)
    fpr_o, tpr_o, _ = roc_curve(y_test, opt_proba)
    axes[1, 1].plot(fpr_b, tpr_b, "b-", lw=2, label=f"Base (AUC={auc(fpr_b, tpr_b):.3f})")
    axes[1, 1].plot(fpr_o, tpr_o, "r-", lw=2, label=f"Optimizado (AUC={auc(fpr_o, tpr_o):.3f})")
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    axes[1, 1].set_title("Curvas ROC")
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3)

    # Importancia de coeficientes
    if hasattr(opt_model, "coef_"):
        coef = opt_model.coef_[0]
        idx = np.argsort(np.abs(coef))[::-1][:10]
        axes[1, 2].barh(
            range(10), coef[idx],
            color=["steelblue" if c > 0 else "lightcoral" for c in coef[idx]],
        )
        axes[1, 2].set_yticks(range(10))
        axes[1, 2].set_yticklabels([f"PC{i+1}" for i in idx])
        axes[1, 2].set_title("Top 10 Componentes más Importantes")
        axes[1, 2].axvline(x=0, color="black", lw=0.5)
        axes[1, 2].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess
    from pca_analysis import run_pca
    from models import split_data

    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, _ = preprocess(raw)
    pca_optimal, X_pca_df, _ = run_pca(X_imputed)

    X_train, X_test, y_train, y_test = split_data(X_pca_df, y)

    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_train, y_train)

    bayes_search, optimized_model = bayesian_optimization(X_train, y_train)
    cv_results = cross_validate(optimized_model, X_train, y_train)
    base_metrics, opt_metrics = compare_models(base_model, optimized_model,
                                               X_train, X_test, y_train, y_test)
