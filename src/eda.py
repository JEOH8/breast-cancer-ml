"""
eda.py
======
Análisis Exploratorio de Datos (EDA) para el dataset de cáncer de mama.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def correlation_matrix(X_imputed: pd.DataFrame, threshold: float = 0.9) -> None:
    """
    Grafica la matriz de correlación y reporta pares con |r| > threshold.

    Parameters
    ----------
    X_imputed : pd.DataFrame
        Características numéricas.
    threshold : float
        Umbral para reportar correlaciones fuertes.
    """
    print("\n" + "=" * 60)
    print("1. MATRIZ DE CORRELACIÓN")
    print("=" * 60)

    corr_matrix = X_imputed.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        vmax=1.0, vmin=-1.0,
        cmap="viridis",
        center=0,
        annot=True,
        annot_kws={"size": 7},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de Correlación de Características", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    # Correlaciones fuertes
    print(f"\n  Correlaciones con |r| > {threshold}:")
    found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                print(
                    f"   {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: "
                    f"r = {corr_matrix.iloc[i, j]:.3f}"
                )
                found = True
    if not found:
        print(f"   No hay correlaciones con |r| > {threshold}")


def boxplots_by_diagnosis(X_imputed: pd.DataFrame, y: pd.Series, n_features: int = 10) -> None:
    """
    Boxplots de las n características con mayor varianza, separadas por diagnóstico.
    Incluye estadísticas descriptivas y test t de Student.

    Parameters
    ----------
    X_imputed : pd.DataFrame
    y : pd.Series
        Variable objetivo binaria (0 = Benigno, 1 = Maligno).
    n_features : int
        Número de características a graficar.
    """
    print(f"\n  Generando boxplots para las {n_features} características con mayor varianza...")

    top_features = X_imputed.var().sort_values(ascending=False).head(n_features).index.tolist()
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.ravel()

    for idx, feature in enumerate(top_features):
        plot_data = pd.DataFrame({
            "Valor": X_imputed[feature],
            "Diagnosis": y.map({0: "Benigno (B)", 1: "Maligno (M)"}),
        })
        sns.boxplot(
            x="Diagnosis", y="Valor", data=plot_data,
            ax=axes[idx], palette=["lightgreen", "lightcoral"],
        )

        benign_vals = X_imputed[y == 0][feature]
        malign_vals = X_imputed[y == 1][feature]

        axes[idx].set_title(f"{feature}\n", fontsize=12)
        axes[idx].text(0.5, 0.95, f"B: μ={benign_vals.mean():.2f}, σ={benign_vals.std():.2f}",
                       transform=axes[idx].transAxes, ha="center", fontsize=9)
        axes[idx].text(0.5, 0.90, f"M: μ={malign_vals.mean():.2f}, σ={malign_vals.std():.2f}",
                       transform=axes[idx].transAxes, ha="center", fontsize=9)

        t_stat, p_val = stats.ttest_ind(benign_vals, malign_vals, equal_var=False)
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        axes[idx].text(0.5, 0.85, f"p={p_val:.2e} {stars}",
                       transform=axes[idx].transAxes, ha="center", fontsize=9,
                       color="red" if p_val < 0.05 else "black")

    plt.suptitle("Distribución de Características por Diagnóstico", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def normality_histograms(X_imputed: pd.DataFrame) -> None:
    """
    Histogramas con KDE y prueba de normalidad Kolmogorov-Smirnov
    para un conjunto representativo de características.

    Parameters
    ----------
    X_imputed : pd.DataFrame
    """
    print("\n  Generando histogramas y pruebas de normalidad...")

    test_features = [
        "radius_mean", "texture_mean", "perimeter_mean",
        "area_mean", "smoothness_mean", "compactness_mean",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, feature in enumerate(test_features):
        sns.histplot(data=X_imputed, x=feature, kde=True, ax=axes[idx], bins=30)
        stat, p_value = stats.kstest(stats.zscore(X_imputed[feature].dropna()), "norm")
        axes[idx].set_title(f"{feature}\nKS p-value = {p_value:.3e}", fontsize=11)
        axes[idx].set_xlabel("")

        normal_text = "Normal" if p_value > 0.05 else "No normal"
        axes[idx].text(
            0.05, 0.95, normal_text,
            transform=axes[idx].transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle("Distribuciones y Pruebas de Normalidad", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def pairplot(data_processed: pd.DataFrame) -> None:
    """
    Pairplot coloreado por diagnóstico.

    Parameters
    ----------
    data_processed : pd.DataFrame
        DataFrame con características + columna 'diagnosis'.
    """
    print("\n  Generando pairplot (puede tardar unos segundos)...")
    sns.set_theme(style="ticks")
    sns.pairplot(data_processed, hue="diagnosis")
    plt.show()


def run_eda(X_imputed: pd.DataFrame, y: pd.Series, data_processed: pd.DataFrame) -> None:
    """
    Ejecuta el EDA completo en orden.
    """
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 60)

    correlation_matrix(X_imputed)
    boxplots_by_diagnosis(X_imputed, y)
    normality_histograms(X_imputed)
    pairplot(data_processed)

    print("\n✓ Análisis exploratorio completado.")


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess

    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, data_processed = preprocess(raw)
    run_eda(X_imputed, y, data_processed)
