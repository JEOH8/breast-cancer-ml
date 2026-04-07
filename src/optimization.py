"""
optimization.py
===============
Optimización Bayesiana de Regresión Logística con BayesSearchCV,
validación cruzada estratificada y comparativa base vs optimizado.

AÑADIDO: Verificación de data leakage para el modelo optimizado
mediante comparativa explícita CV-10 vs holdout test.
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
    optimized_model : LogisticRegression reentrenado en todo el train
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
    Validación cruzada estratificada 10-fold para el modelo optimizado.

    Returns
    -------
    cv_results : dict[str, np.ndarray]
    """
    print("\n" + "=" * 70)
    print("VALIDACIÓN CRUZADA DEL MODELO OPTIMIZADO (10-fold)")
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


# ── Verificación de leakage: CV vs holdout ─────────────────────────────────

def verify_no_leakage_optimized(cv_results: dict,
                                 opt_metrics: dict,
                                 base_metrics: dict) -> None:
    """
    Compara métricas de CV-10 vs holdout test para el modelo optimizado
    y el modelo base, con visualización clara de diferencias.

    Un modelo sin data leakage mostrará diferencias pequeñas (<3%)
    entre CV y holdout en todas las métricas.

    Parameters
    ----------
    cv_results  : dict de cross_validate()
    opt_metrics : dict con métricas del modelo optimizado en holdout
    base_metrics: dict con métricas del modelo base en holdout
    """
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE DATA LEAKAGE — MODELOS FINALES DE REGRESIÓN LOGÍSTICA")
    print("=" * 70)

    metrics_map = {
        "accuracy": "Accuracy Test",
        "precision": "Precision Test",
        "recall":    "Recall Test",
        "f1":        "F1 Test",
        "roc_auc":   "AUC Test",
    }

    rows = []
    for cv_key, holdout_key in metrics_map.items():
        cv_mean = cv_results[cv_key].mean()
        holdout = opt_metrics[holdout_key]
        diff = holdout - cv_mean
        status = "✅ Sin leakage" if abs(diff) < 0.02 else \
                 "⚠️ Revisar"    if abs(diff) < 0.05 else \
                 "❌ Alerta"
        rows.append({
            "Métrica":         cv_key.upper(),
            "CV-10 (train)":   f"{cv_mean:.3%}",
            "Holdout (test)":  f"{holdout:.3%}",
            "Diferencia":      f"{diff:+.3%}",
            "Estado":          status,
        })
        print(f"  {cv_key.upper():12s}: CV={cv_mean:.3%} | Holdout={holdout:.3%} "
              f"| Δ={diff:+.3%} | {status}")

    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # Gráfico comparativo
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metric_labels = [r["Métrica"] for r in rows]
    cv_vals      = [cv_results[k].mean() for k in metrics_map]
    holdout_vals = [opt_metrics[v] for v in metrics_map.values()]
    x = np.arange(len(metric_labels))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, cv_vals,      width,
                        label="CV-10 (train)",    color="steelblue",  alpha=0.85)
    bars2 = axes[0].bar(x + width/2, holdout_vals, width,
                        label="Holdout (test)", color="lightcoral", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0.88, 1.02)
    axes[0].set_ylabel("Valor de la Métrica")
    axes[0].set_title("Reg. Logística Optimizada\nCV-10 vs Holdout Test")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    for i, (cv_v, ho_v) in enumerate(zip(cv_vals, holdout_vals)):
        diff = ho_v - cv_v
        color = "green" if abs(diff) < 0.02 else \
                "orange" if abs(diff) < 0.05 else "red"
        axes[0].annotate(f"{diff:+.2%}",
                         xy=(x[i], max(cv_v, ho_v) + 0.004),
                         ha="center", fontsize=9,
                         color=color, fontweight="bold")

    # Boxplot de CV con línea del holdout
    cv_plot_data = [cv_results[k] for k in metrics_map]
    bp = axes[1].boxplot(cv_plot_data, labels=metric_labels,
                         patch_artist=True,
                         boxprops=dict(facecolor="steelblue", alpha=0.6))
    for i, ho_v in enumerate(holdout_vals):
        axes[1].plot(i + 1, ho_v, "r*", markersize=14,
                     label="Holdout" if i == 0 else "")
    axes[1].set_ylabel("Valor de la Métrica")
    axes[1].set_title("Distribución CV-10 por Fold\n(★ = valor holdout test)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0.80, 1.05)

    plt.suptitle("Verificación de Data Leakage — Reg. Logística Optimizada\n"
                 "Diferencias pequeñas confirman ausencia de leakage significativo",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    # Interpretación textual
    max_diff = max(abs(opt_metrics[v] - cv_results[k].mean())
                   for k, v in metrics_map.items())
    print(f"\n  Diferencia máxima observada: {max_diff:.2%}")
    if max_diff < 0.02:
        print("  ✅ CONCLUSIÓN: Sin data leakage significativo detectado.")
        print("     Los resultados del holdout son consistentes con CV.")
    elif max_diff < 0.05:
        print("  ⚠️  CONCLUSIÓN: Diferencia leve. Monitorear pero no crítico.")
    else:
        print("  ❌ CONCLUSIÓN: Diferencia elevada. Investigar posible leakage.")


# ── Comparativa base vs optimizado ─────────────────────────────────────────

def compare_models(base_model, optimized_model, X_train, X_test, y_train, y_test):
    """
    Genera métricas y visualizaciones comparando base vs optimizado.

    Returns
    -------
    base_metrics, opt_metrics : dict, dict
    """
    def compute_metrics(model, X_tr, X_te, y_tr, y_te):
        y_tr_pred  = model.predict(X_tr)
        y_te_pred  = model.predict(X_te)
        y_te_proba = (model.predict_proba(X_te)[:, 1]
                      if hasattr(model, "predict_proba") else y_te_pred)
        return {
            "Accuracy Train": accuracy_score(y_tr, y_tr_pred),
            "Accuracy Test":  accuracy_score(y_te, y_te_pred),
            "Precision Test": precision_score(y_te, y_te_pred),
            "Recall Test":    recall_score(y_te, y_te_pred),
            "F1 Test":        f1_score(y_te, y_te_pred),
            "AUC Test":       roc_auc_score(y_te, y_te_proba),
        }, y_te_pred, y_te_proba

    base_metrics, base_pred, base_proba = compute_metrics(
        base_model, X_train, X_test, y_train, y_test)
    opt_metrics, opt_pred, opt_proba = compute_metrics(
        optimized_model, X_train, X_test, y_train, y_test)

    rows = []
    for metric in base_metrics:
        base_val = base_metrics[metric]
        opt_val  = opt_metrics[metric]
        rows.append({
            "Métrica":           metric,
            "Modelo Base":       f"{base_val:.3%}",
            "Modelo Optimizado": f"{opt_val:.3%}",
            "Mejora Absoluta":   f"{opt_val - base_val:+.3%}",
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    _plot_comparison(
        base_metrics, opt_metrics,
        confusion_matrix(y_test, base_pred),
        confusion_matrix(y_test, opt_pred),
        base_proba, opt_proba, y_test, optimized_model,
    )

    return base_metrics, opt_metrics


# ── Función de predicción ───────────────────────────────────────────────────

def predict_breast_cancer(features, model, threshold: float = 0.5) -> dict:
    features = np.array(features).reshape(1, -1)
    expected = model.n_features_in_
    if features.shape[1] != expected:
        raise ValueError(f"Se esperaban {expected} características, "
                         f"recibidas {features.shape[1]}.")

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
            if label == "Maligno" else "SEGUIMIENTO RUTINARIO"
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _plot_cv_results(cv_results: dict, n_splits: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cv_data = [{"Métrica": m, "Score": s}
               for m, scores in cv_results.items() for s in scores]
    cv_df = pd.DataFrame(cv_data)

    sns.boxplot(x="Métrica", y="Score", data=cv_df, ax=axes[0])
    sns.stripplot(x="Métrica", y="Score", data=cv_df,
                  color="black", alpha=0.5, ax=axes[0])
    axes[0].set_title("Distribución de Métricas en Validación Cruzada")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")

    folds = np.arange(1, n_splits + 1)
    for metric in ["accuracy", "f1", "roc_auc"]:
        if metric in cv_results:
            axes[1].plot(folds, cv_results[metric],
                         marker="o", label=metric, linewidth=2)
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

    for i, (label, metrics, color) in enumerate(zip(
        ["Base", "Optimizado"], [base_m, opt_m], ["skyblue", "lightcoral"]
    )):
        vals = [metrics["Accuracy Train"], metrics["Accuracy Test"]]
        axes[0, 0].bar(np.arange(2) + i * width, vals, width,
                       label=label, alpha=0.8, color=color)
    axes[0, 0].set_xticks(np.arange(2) + width / 2)
    axes[0, 0].set_xticklabels(["Entrenamiento", "Prueba"])
    axes[0, 0].set_title("Comparación de Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    keys = ["Precision Test", "Recall Test", "F1 Test"]
    for i, (label, metrics, color) in enumerate(zip(
        ["Base", "Optimizado"], [base_m, opt_m], ["skyblue", "lightcoral"]
    )):
        vals = [metrics[k] for k in keys]
        axes[0, 1].bar(np.arange(3) + i * width, vals, width,
                       label=label, alpha=0.8, color=color)
    axes[0, 1].set_xticks(np.arange(3) + width / 2)
    axes[0, 1].set_xticklabels(["Precision", "Recall", "F1"])
    axes[0, 1].set_title("Métricas de Clasificación (Test)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[0, 2],
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"])
    axes[0, 2].set_title("Confusión — Modelo Base")

    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", ax=axes[1, 0],
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"])
    axes[1, 0].set_title("Confusión — Modelo Optimizado")

    fpr_b, tpr_b, _ = roc_curve(y_test, base_proba)
    fpr_o, tpr_o, _ = roc_curve(y_test, opt_proba)
    axes[1, 1].plot(fpr_b, tpr_b, "b-", lw=2,
                    label=f"Base (AUC={auc(fpr_b, tpr_b):.3f})")
    axes[1, 1].plot(fpr_o, tpr_o, "r-", lw=2,
                    label=f"Optimizado (AUC={auc(fpr_o, tpr_o):.3f})")
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    axes[1, 1].set_title("Curvas ROC")
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3)

    if hasattr(opt_model, "coef_"):
        coef = opt_model.coef_[0]
        idx  = np.argsort(np.abs(coef))[::-1][:10]
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
    raw  = load_data(path)
    X_imputed, y, _ = preprocess(raw)
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_imputed, y)
    pca_optimal, X_train_pca, X_test_pca, scaler = run_pca(X_train_raw, X_test_raw)

    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_train_pca, y_train)

    bayes_search, optimized_model = bayesian_optimization(X_train_pca, y_train)
    cv_results  = cross_validate(optimized_model, X_train_pca, y_train)
    base_metrics, opt_metrics = compare_models(
        base_model, optimized_model,
        X_train_pca, X_test_pca, y_train, y_test
    )
    verify_no_leakage_optimized(cv_results, opt_metrics, base_metrics)
