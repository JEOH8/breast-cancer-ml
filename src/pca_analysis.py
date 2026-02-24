"""
pca_analysis.py
===============
Análisis de Componentes Principales (PCA) para reducción de dimensionalidad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(X_imputed: pd.DataFrame, n_components_optimal: int = 17):
    """
    Ejecuta el pipeline completo de PCA:
      1. Normalización estándar.
      2. PCA exploratorio (todos los componentes) + gráficos de varianza.
      3. PCA con n_components_optimal componentes.
      4. Visualización en 2D con elipses de confianza.
      5. Análisis de cargas (loadings).

    Parameters
    ----------
    X_imputed : pd.DataFrame
        Características sin escalar.
    n_components_optimal : int
        Número de componentes a retener en el modelo final.

    Returns
    -------
    pca_optimal : PCA
        Objeto PCA ajustado.
    X_pca_df : pd.DataFrame
        Datos transformados al espacio PCA.
    scaler : StandardScaler
        Scaler ajustado (útil para transformar nuevas muestras).
    """
    print("=" * 60)
    print("ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
    print("=" * 60)

    # ── 1. Normalización ───────────────────────────────────────────
    print("\n1. NORMALIZACIÓN DE DATOS...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"   Shape: {X_scaled.shape} | Media≈{X_scaled.mean():.2f} | Std≈{X_scaled.std():.2f}")

    # ── 2. PCA exploratorio ────────────────────────────────────────
    print("\n2. PCA EXPLORATORIO (todos los componentes)...")
    pca_full = PCA()
    pca_full.fit_transform(X_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    _plot_variance(explained_variance, cumulative_variance)

    # Umbrales
    for pct in [0.80, 0.90, 0.95, 0.99]:
        n = np.argmax(cumulative_variance >= pct) + 1
        print(f"   Componentes para {pct:.0%} de varianza: {n}")

    # ── 3. PCA óptimo ─────────────────────────────────────────────
    print(f"\n3. PCA CON {n_components_optimal} COMPONENTES (99% varianza)...")
    pca_optimal = PCA(n_components=n_components_optimal)
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)
    print(f"   Varianza explicada: {pca_optimal.explained_variance_ratio_.sum():.3%}")
    print(f"   Shape transformado: {X_pca_optimal.shape}")

    # ── 4. Visualización 2D ───────────────────────────────────────
    _plot_2d_pca(X_pca_optimal, pca_optimal)

    # ── 5. Cargas (loadings) ──────────────────────────────────────
    loadings = pd.DataFrame(
        pca_optimal.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components_optimal)],
        index=X_imputed.columns,
    )
    print("\n4. TOP 10 VARIABLES — PC1:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    print("\n   TOP 10 VARIABLES — PC2:")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    # ── 6. Resultado final ────────────────────────────────────────
    X_pca_df = pd.DataFrame(
        X_pca_optimal,
        columns=[f"PC{i+1}" for i in range(n_components_optimal)],
    )

    print("\n" + "=" * 60)
    print("RESUMEN PCA")
    print("=" * 60)
    print(f"  • Componentes seleccionados : {n_components_optimal}")
    print(f"  • Varianza explicada        : {pca_optimal.explained_variance_ratio_.sum():.2%}")
    print(f"  • Reducción dimensionalidad : {X_imputed.shape[1]} → {n_components_optimal}")
    print(f"  • Tasa de compresión        : {(1 - n_components_optimal / X_imputed.shape[1]):.1%}")
    print("=" * 60)
    print("\n✓ PCA completado. Datos listos para modelos.")

    return pca_optimal, X_pca_df, scaler


# ── Helpers de visualización ───────────────────────────────────────────────

def _plot_variance(explained: np.ndarray, cumulative: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.bar(range(1, len(explained) + 1), explained, alpha=0.6)
    ax1.plot(range(1, len(explained) + 1), explained, "r-", marker="o")
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("Varianza Explicada")
    ax1.set_title("Varianza Explicada por Componente")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(cumulative) + 1), cumulative, "b-", marker="o", linewidth=2)
    ax2.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95%")
    ax2.axhline(y=0.90, color="g", linestyle="--", alpha=0.7, label="90%")
    ax2.set_xlabel("Número de Componentes")
    ax2.set_ylabel("Varianza Acumulada")
    ax2.set_title("Varianza Acumulada")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_2d_pca(X_pca: np.ndarray, pca: PCA) -> None:
    # y no está disponible aquí; se usa un placeholder genérico
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50, edgecolors="k")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
    plt.title("Primeras Dos Componentes Principales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_2d_with_labels(X_pca_df: pd.DataFrame, y: pd.Series, pca: PCA) -> None:
    """
    Visualiza las primeras 2 componentes coloreadas por diagnóstico,
    con elipses de confianza por clase.

    Parameters
    ----------
    X_pca_df : pd.DataFrame
        Datos transformados.
    y : pd.Series
        Etiquetas de clase.
    pca : PCA
        Objeto PCA ajustado.
    """
    X_arr = X_pca_df.values
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y.values,
                          cmap="coolwarm", alpha=0.7, s=50, edgecolors="k")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
    plt.title("Primeras Dos Componentes Principales")
    plt.colorbar(scatter, label="Diagnóstico (0=Benigno, 1=Maligno)")
    plt.grid(True, alpha=0.3)

    for diag_val in [0, 1]:
        mask = y.values == diag_val
        if mask.sum() > 1:
            cov = np.cov(X_arr[mask, 0], X_arr[mask, 1])
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(
                xy=(X_arr[mask, 0].mean(), X_arr[mask, 1].mean()),
                width=lambda_[0] * 4, height=lambda_[1] * 4,
                angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                alpha=0.2, color="red" if diag_val == 1 else "green",
            )
            plt.gca().add_patch(ell)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess

    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, data_processed = preprocess(raw)
    pca_optimal, X_pca_df, scaler = run_pca(X_imputed)
    plot_2d_with_labels(X_pca_df, y, pca_optimal)
