"""
models.py
=========
Entrenamiento, evaluación, validación cruzada y comparativa
de modelos de clasificación con PCA.

MEJORA: Validación cruzada aplicada a los 5 modelos base para
comparación honesta con los resultados del conjunto de prueba.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)


# ── Split de datos ─────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    División estratificada. Debe llamarse ANTES de run_pca() para
    evitar data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
        shuffle=True,
    )

    print("DIVISIÓN DE DATOS (antes del escalado/PCA):")
    print(f"  Entrenamiento : {X_train.shape[0]} muestras")
    print(f"  Prueba        : {X_test.shape[0]} muestras")

    for name, y_split in [("Entrenamiento", y_train), ("Prueba", y_test)]:
        print(f"\n  Distribución en {name}:")
        print(f"    Benigno: {(y_split == 0).sum()} ({(y_split == 0).mean():.2%})")
        print(f"    Maligno: {(y_split == 1).sum()} ({(y_split == 1).mean():.2%})")

    return X_train, X_test, y_train, y_test


# ── Evaluación individual ──────────────────────────────────────────────────

def evaluar_modelo_clasificacion(modelo, nombre_modelo,
                                  X_train, X_test, y_train, y_test):
    """Entrena y evalúa un modelo. Devuelve métricas y modelo entrenado."""
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)
    y_pred_proba = (
        modelo.predict_proba(X_test)[:, 1]
        if hasattr(modelo, "predict_proba") else None
    )

    metrics = {
        "Nombre": nombre_modelo,
        "Accuracy Train": accuracy_score(y_train, y_pred_train),
        "Accuracy Test":  accuracy_score(y_test,  y_pred_test),
        "Precision Test": precision_score(y_test, y_pred_test),
        "Recall Test":    recall_score(y_test,    y_pred_test),
        "F1 Test":        f1_score(y_test,         y_pred_test),
        "AUC Test": (
            roc_auc_score(y_test, y_pred_proba)
            if y_pred_proba is not None else np.nan
        ),
    }

    cm = confusion_matrix(y_test, y_pred_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Benigno", "Maligno"],
                yticklabels=["Benigno", "Maligno"])
    axes[0].set_xlabel("Predicho"); axes[0].set_ylabel("Real")
    axes[0].set_title(f"{nombre_modelo} — Matriz de Confusión")

    report = classification_report(y_test, y_pred_test, output_dict=False)
    axes[1].text(0.1, 0.9, report, fontfamily="monospace", fontsize=10,
                 verticalalignment="top", transform=axes[1].transAxes)
    axes[1].axis("off")
    axes[1].set_title(f"{nombre_modelo} — Reporte de Clasificación")

    plt.tight_layout(); plt.show()
    return metrics, modelo


# ── Validación cruzada para todos los modelos ──────────────────────────────

def cross_validate_all_models(X_train, y_train, n_splits: int = 10):
    """
    Ejecuta validación cruzada estratificada (10-fold) para los 5
    clasificadores base y devuelve un DataFrame comparativo.

    Permite detectar si los resultados del holdout test son coherentes
    con los de CV (inconsistencia → posible data leakage u overfitting).

    Returns
    -------
    cv_summary : pd.DataFrame
    """
    print("\n" + "=" * 70)
    print("VALIDACIÓN CRUZADA — 5 MODELOS BASE (10-fold estratificado)")
    print("=" * 70)

    classifiers = [
        (LogisticRegression(random_state=42, max_iter=1000), "Regresión Logística"),
        (KNeighborsClassifier(n_neighbors=5),                "KNN (k=5)"),
        (SVC(kernel="rbf", probability=True, random_state=42), "SVM (RBF)"),
        (DecisionTreeClassifier(random_state=42, max_depth=5), "Árbol de Decisión"),
        (RandomForestClassifier(random_state=42, n_estimators=100), "Random Forest"),
    ]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for clf, nombre in classifiers:
        f1_scores  = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1",       n_jobs=-1)
        acc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        rec_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="recall",   n_jobs=-1)
        auc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc",  n_jobs=-1)

        rows.append({
            "Modelo":       nombre,
            "Accuracy CV":  f"{acc_scores.mean():.3%} ± {acc_scores.std():.3%}",
            "Recall CV":    f"{rec_scores.mean():.3%} ± {rec_scores.std():.3%}",
            "F1 CV":        f"{f1_scores.mean():.3%} ± {f1_scores.std():.3%}",
            "AUC CV":       f"{auc_scores.mean():.3f} ± {auc_scores.std():.3f}",
            # Guardar medias para comparativa posterior
            "_acc_mean":    acc_scores.mean(),
            "_f1_mean":     f1_scores.mean(),
            "_rec_mean":    rec_scores.mean(),
            "_auc_mean":    auc_scores.mean(),
        })
        print(f"\n  {nombre}:")
        print(f"    Accuracy : {acc_scores.mean():.3%} ± {acc_scores.std():.3%}")
        print(f"    Recall   : {rec_scores.mean():.3%} ± {rec_scores.std():.3%}")
        print(f"    F1       : {f1_scores.mean():.3%} ± {f1_scores.std():.3%}")
        print(f"    AUC      : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    cv_summary = pd.DataFrame(rows)
    return cv_summary


# ── Comparativa CV vs Holdout ──────────────────────────────────────────────

def plot_cv_vs_holdout(cv_summary: pd.DataFrame,
                       holdout_results: list,
                       metric: str = "F1") -> None:
    """
    Gráfico de barras comparando métricas de CV (train) vs holdout (test)
    para detectar overfitting o data leakage.

    Parameters
    ----------
    cv_summary     : DataFrame de cross_validate_all_models()
    holdout_results: lista de dicts de evaluar_modelo_clasificacion()
    metric         : "F1", "Accuracy" o "Recall"
    """
    metric_map = {
        "F1":       ("_f1_mean",  "F1 Test"),
        "Accuracy": ("_acc_mean", "Accuracy Test"),
        "Recall":   ("_rec_mean", "Recall Test"),
    }
    cv_col, holdout_col = metric_map[metric]

    holdout_df = pd.DataFrame(holdout_results)
    nombres = cv_summary["Modelo"].tolist()

    cv_vals      = cv_summary[cv_col].values
    holdout_vals = holdout_df.set_index("Nombre").loc[nombres, holdout_col].values

    x = np.arange(len(nombres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, cv_vals,      width, label=f"{metric} CV (train)",   color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width/2, holdout_vals, width, label=f"{metric} Holdout (test)", color="lightcoral", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0.8, 1.02)
    ax.set_title(f"Comparativa {metric}: Validación Cruzada vs Conjunto de Prueba\n"
                 "(Diferencias pequeñas → sin data leakage significativo)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Anotar diferencias
    for i, (cv_v, ho_v) in enumerate(zip(cv_vals, holdout_vals)):
        diff = ho_v - cv_v
        color = "green" if abs(diff) < 0.02 else "orange" if abs(diff) < 0.05 else "red"
        ax.annotate(f"{diff:+.2%}", xy=(x[i], max(cv_v, ho_v) + 0.005),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Interpretación
    print("\nINTERPRETACIÓN (diferencia CV → Holdout):")
    print("  Verde  (<2%)  : sin overfitting ni leakage aparente")
    print("  Naranja (2-5%): revisar, posible overfitting leve")
    print("  Rojo   (>5%)  : señal de alerta — investigar leakage u overfitting")


# ── Entrenamiento de todos los modelos ─────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    """Entrena cinco clasificadores y devuelve resultados y modelos."""
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN")
    print("=" * 60)

    classifiers = [
        (LogisticRegression(random_state=42, max_iter=1000), "Regresión Logística"),
        (KNeighborsClassifier(n_neighbors=5),                "KNN (k=5)"),
        (SVC(kernel="rbf", probability=True, random_state=42), "SVM (RBF)"),
        (DecisionTreeClassifier(random_state=42, max_depth=5), "Árbol de Decisión"),
        (RandomForestClassifier(random_state=42, n_estimators=100), "Random Forest"),
    ]

    resultados, models_dict = [], {}

    for clf, nombre in classifiers:
        print(f"\n  Entrenando: {nombre}...")
        metrics, trained_model = evaluar_modelo_clasificacion(
            clf, nombre, X_train, X_test, y_train, y_test
        )
        resultados.append(metrics)
        models_dict[nombre] = trained_model

    print("\n✓ Todos los modelos entrenados y evaluados.")
    return resultados, models_dict


def plot_comparison(resultados, models_dict, X_test, y_test, pca_optimal):
    """Visualización comparativa con curvas ROC mejoradas."""
    resultados_df = pd.DataFrame(resultados).sort_values("Accuracy Test", ascending=False)

    print("\nTABLA COMPARATIVA DE MODELOS (Holdout Test Set):")
    print(resultados_df.drop(columns=[c for c in resultados_df.columns
                                       if c.startswith("_")]).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    x_pos = np.arange(len(resultados_df))
    width = 0.35

    # Accuracy train vs test
    axes[0, 0].bar(x_pos - width/2, resultados_df["Accuracy Train"],
                   width, label="Train", alpha=0.8, color="skyblue")
    axes[0, 0].bar(x_pos + width/2, resultados_df["Accuracy Test"],
                   width, label="Test",  alpha=0.8, color="lightcoral")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(resultados_df["Nombre"], rotation=45, ha="right")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Comparación de Accuracy entre Modelos")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Precision / Recall / F1
    for i, metric in enumerate(["Precision Test", "Recall Test", "F1 Test"]):
        axes[0, 1].bar(x_pos + i*0.25, resultados_df[metric],
                       width=0.25, label=metric.replace(" Test", ""))
    axes[0, 1].set_xticks(x_pos + 0.25)
    axes[0, 1].set_xticklabels(resultados_df["Nombre"], rotation=45, ha="right")
    axes[0, 1].set_ylabel("Valor")
    axes[0, 1].set_title("Métricas de Clasificación (Test Set)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Curvas ROC comparativas (mejoradas)
    axes[1, 0].plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio (AUC=0.5)")
    colors = ["royalblue", "forestgreen", "crimson", "darkorchid", "darkorange"]
    for (nombre, model), color in zip(models_dict.items(), colors):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc_val = roc_auc_score(y_test, proba)
            axes[1, 0].plot(fpr, tpr, lw=2,
                            label=f"{nombre} (AUC={auc_val:.3f})", color=color)
    axes[1, 0].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    axes[1, 0].set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    axes[1, 0].set_title("Curvas ROC — Clasificación Binaria\n"
                          "(Maligno vs Benigno)")
    axes[1, 0].legend(loc="lower right", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect("equal")

    # Info resumen
    best = resultados_df.iloc[0]
    info = (
        f"RESUMEN — SIN DATA LEAKAGE\n\n"
        f"Pipeline corregido:\n"
        f"  Split → Scaler(fit train) → PCA(fit train)\n\n"
        f"Componentes PCA : {X_test.shape[1]}\n"
        f"Varianza expl.  : {pca_optimal.explained_variance_ratio_.sum():.1%}\n"
        f"Mejor modelo    : {best['Nombre']}\n"
        f"Accuracy (test) : {best['Accuracy Test']:.1%}\n"
        f"Mejor F1        : {resultados_df['F1 Test'].max():.3f}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.95, info, fontsize=11, verticalalignment="top",
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.7))

    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess
    from pca_analysis import run_pca

    path = download_dataset()
    raw  = load_data(path)
    X_imputed, y, _ = preprocess(raw)

    # ✓ Split ANTES del PCA
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_imputed, y)
    pca_optimal, X_train_pca, X_test_pca, scaler = run_pca(X_train_raw, X_test_raw)

    cv_summary = cross_validate_all_models(X_train_pca, y_train)
    resultados, models_dict = train_all_models(X_train_pca, X_test_pca, y_train, y_test)
    plot_cv_vs_holdout(cv_summary, resultados, metric="F1")
    plot_comparison(resultados, models_dict, X_test_pca, y_test, pca_optimal)
