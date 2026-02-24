"""
models.py
=========
Entrenamiento, evaluación y comparativa de modelos de clasificación con PCA.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)


# ── Split de datos ─────────────────────────────────────────────────────────

def split_data(X_pca_df: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    División estratificada en conjuntos de entrenamiento y prueba.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca_df, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
        shuffle=True,
    )

    print("DIVISIÓN DE DATOS:")
    print(f"  Entrenamiento : {X_train.shape[0]} muestras")
    print(f"  Prueba        : {X_test.shape[0]} muestras")

    for name, y_split in [("Entrenamiento", y_train), ("Prueba", y_test)]:
        print(f"\n  Distribución en {name}:")
        print(f"    Benigno: {(y_split == 0).sum()} ({(y_split == 0).mean():.2%})")
        print(f"    Maligno: {(y_split == 1).sum()} ({(y_split == 1).mean():.2%})")

    return X_train, X_test, y_train, y_test


# ── Evaluación individual ──────────────────────────────────────────────────

def evaluar_modelo_clasificacion(modelo, nombre_modelo, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo de clasificación. Muestra matriz de confusión
    y reporte de clasificación.

    Returns
    -------
    metrics : dict
        Métricas clave del modelo.
    modelo : estimador ajustado
    """
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_proba = (
        modelo.predict_proba(X_test)[:, 1]
        if hasattr(modelo, "predict_proba")
        else None
    )

    metrics = {
        "Nombre": nombre_modelo,
        "Accuracy Train": accuracy_score(y_train, y_pred_train),
        "Accuracy Test": accuracy_score(y_test, y_pred_test),
        "Precision Test": precision_score(y_test, y_pred_test),
        "Recall Test": recall_score(y_test, y_pred_test),
        "F1 Test": f1_score(y_test, y_pred_test),
        "AUC Test": (
            roc_auc_score(y_test, y_pred_proba)
            if y_pred_proba is not None
            else np.nan
        ),
    }

    cm = confusion_matrix(y_test, y_pred_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_xlabel("Predicho")
    axes[0].set_ylabel("Real")
    axes[0].set_title(f"{nombre_modelo} — Matriz de Confusión")

    report = classification_report(y_test, y_pred_test, output_dict=False)
    axes[1].text(0.1, 0.9, report, fontfamily="monospace", fontsize=10,
                 verticalalignment="top", transform=axes[1].transAxes)
    axes[1].axis("off")
    axes[1].set_title(f"{nombre_modelo} — Reporte de Clasificación")

    plt.tight_layout()
    plt.show()

    return metrics, modelo


# ── Comparativa final ──────────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Entrena cinco clasificadores y devuelve sus resultados.

    Returns
    -------
    resultados : list[dict]
    models_dict : dict
        Diccionario nombre → estimador ajustado.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN")
    print("=" * 60)

    classifiers = [
        (LogisticRegression(random_state=42, max_iter=1000), "Regresión Logística"),
        (KNeighborsClassifier(n_neighbors=5), "KNN (k=5)"),
        (SVC(kernel="rbf", probability=True, random_state=42), "SVM (RBF)"),
        (DecisionTreeClassifier(random_state=42, max_depth=5), "Árbol de Decisión"),
        (RandomForestClassifier(random_state=42, n_estimators=100), "Random Forest"),
    ]

    resultados = []
    models_dict = {}

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
    """
    Visualización comparativa: accuracy, métricas de test, curvas ROC e info PCA.

    Parameters
    ----------
    resultados : list[dict]
    models_dict : dict
    X_test : array-like
    y_test : array-like
    pca_optimal : PCA
    """
    resultados_df = pd.DataFrame(resultados).sort_values("Accuracy Test", ascending=False)

    print("\nTABLA COMPARATIVA DE MODELOS:")
    print(resultados_df.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    x_pos = np.arange(len(resultados_df))
    width = 0.35

    # Accuracy
    axes[0, 0].bar(x_pos - width / 2, resultados_df["Accuracy Train"],
                   width, label="Train", alpha=0.8, color="skyblue")
    axes[0, 0].bar(x_pos + width / 2, resultados_df["Accuracy Test"],
                   width, label="Test", alpha=0.8, color="lightcoral")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(resultados_df["Nombre"], rotation=45, ha="right")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Comparación de Accuracy entre Modelos")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision / Recall / F1
    for i, metric in enumerate(["Precision Test", "Recall Test", "F1 Test"]):
        axes[0, 1].bar(x_pos + i * 0.25, resultados_df[metric],
                       width=0.25, label=metric.replace(" Test", ""))
    axes[0, 1].set_xticks(x_pos + 0.25)
    axes[0, 1].set_xticklabels(resultados_df["Nombre"], rotation=45, ha="right")
    axes[0, 1].set_ylabel("Valor")
    axes[0, 1].set_title("Métricas de Clasificación (Test)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Curvas ROC
    axes[1, 0].plot([0, 1], [0, 1], "k--", label="Aleatorio")
    colors = ["blue", "green", "red", "purple", "orange"]
    for (nombre, model), color in zip(models_dict.items(), colors):
        if hasattr(model, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auc_val = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            axes[1, 0].plot(fpr, tpr, label=f"{nombre} (AUC={auc_val:.3f})", color=color)
    axes[1, 0].set_xlabel("Tasa de Falsos Positivos")
    axes[1, 0].set_ylabel("Tasa de Verdaderos Positivos")
    axes[1, 0].set_title("Curvas ROC Comparativas")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)

    # Info PCA
    best = resultados_df.iloc[0]
    info = (
        f"RESUMEN PCA + CLASIFICACIÓN\n\n"
        f"Componentes PCA : {X_test.shape[1]}\n"
        f"Varianza expl.  : {pca_optimal.explained_variance_ratio_.sum():.1%}\n"
        f"Mejor modelo    : {best['Nombre']}\n"
        f"Accuracy (test) : {best['Accuracy Test']:.1%}\n"
        f"Mejor F1        : {resultados_df['F1 Test'].max():.3f}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.1, 0.9, info, fontsize=12, verticalalignment="top",
        transform=axes[1, 1].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess
    from pca_analysis import run_pca

    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, _ = preprocess(raw)
    pca_optimal, X_pca_df, _ = run_pca(X_imputed)

    X_train, X_test, y_train, y_test = split_data(X_pca_df, y)
    resultados, models_dict = train_all_models(X_train, X_test, y_train, y_test)
    plot_comparison(resultados, models_dict, X_test, y_test, pca_optimal)
