"""
pca_analysis.py
===============
Análisis de Componentes Principales (PCA) para reducción de dimensionalidad.

CORRECCIÓN DE DATA LEAKAGE:
    El StandardScaler y el PCA se ajustan ÚNICAMENTE sobre X_train.
    X_test se transforma con los parámetros aprendidos del train,
    sin que sus valores influyan en el ajuste.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def run_pca(X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            n_components_optimal: int = 17):
    """
    Pipeline completo de PCA sin data leakage:
      1. Imputa valores nulos (fit solo en train)
      2. Normalización estándar (fit solo en train)
      3. PCA exploratorio sobre train
      4. PCA con n_components_optimal (fit solo en train)
      5. Transform de train y test por separado
      6. Visualizaciones y análisis de cargas

    Parameters
    ----------
    X_train : pd.DataFrame  — conjunto de entrenamiento sin escalar
    X_test  : pd.DataFrame  — conjunto de prueba sin escalar
    n_components_optimal : int

    Returns
    -------
    pca_optimal : PCA ajustado sobre train
    X_train_pca : pd.DataFrame — train transformado
    X_test_pca  : pd.DataFrame — test transformado
    scaler      : StandardScaler ajustado sobre train
    """
    print("=" * 60)
    print("ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
    print("(Ajuste exclusivo sobre conjunto de entrenamiento)")
    print("=" * 60)

    # ── 1. Imputación (fit solo en train) ────────────────────────────
    if X_train.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=X_train.columns
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test), columns=X_test.columns
        )
        print("\n  Imputación aplicada (fit en train, transform en test)")
    else:
        X_train_imp = X_train.copy()
        X_test_imp  = X_test.copy()
        print("\n  ✓ Sin valores nulos — imputación no necesaria")

    # ── 2. Normalización (fit solo en train) ─────────────────────────
    print("\n1. NORMALIZACIÓN DE DATOS (fit en train)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)
    print(f"   Train — Media≈{X_train_scaled.mean():.2f} | Std≈{X_train_scaled.std():.2f}")
    print(f"   Test  — Media≈{X_test_scaled.mean():.2f}  | Std≈{X_test_scaled.std():.2f}")

    # ── 3. PCA exploratorio (sobre train) ────────────────────────────
    print("\n2. PCA EXPLORATORIO (todos los componentes, solo train)...")
    pca_full = PCA()
    pca_full.fit(X_train_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    _plot_variance(explained_variance, cumulative_variance)

    for pct in [0.80, 0.90, 0.95, 0.99]:
        n = np.argmax(cumulative_variance >= pct) + 1
        print(f"   Componentes para {pct:.0%} de varianza: {n}")

    # ── 4. PCA óptimo (fit en train, transform en ambos) ─────────────
    print(f"\n3. PCA CON {n_components_optimal} COMPONENTES (99% varianza)...")
    pca_optimal = PCA(n_components=n_components_optimal)
    X_train_pca_arr = pca_optimal.fit_transform(X_train_scaled)
    X_test_pca_arr  = pca_optimal.transform(X_test_scaled)

    cols = [f"PC{i+1}" for i in range(n_components_optimal)]
    X_train_pca = pd.DataFrame(X_train_pca_arr, columns=cols,
                                index=X_train.index)
    X_test_pca  = pd.DataFrame(X_test_pca_arr,  columns=cols,
                                index=X_test.index)

    print(f"   Varianza explicada : {pca_optimal.explained_variance_ratio_.sum():.3%}")
    print(f"   Shape train        : {X_train_pca.shape}")
    print(f"   Shape test         : {X_test_pca.shape}")

    # ── 5. Cargas ─────────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca_optimal.components_.T,
        columns=cols,
        index=X_train_imp.columns,
    )
    print("\n4. TOP 10 VARIABLES — PC1:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    print("\n   TOP 10 VARIABLES — PC2:")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    # ── 6. Resumen ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN PCA")
    print("=" * 60)
    print(f"  • Componentes seleccionados : {n_components_optimal}")
    print(f"  • Varianza explicada        : {pca_optimal.explained_variance_ratio_.sum():.2%}")
    print(f"  • Reducción dimensionalidad : {X_train_imp.shape[1]} → {n_components_optimal}")
    print(f"  • Tasa de compresión        : {(1 - n_components_optimal / X_train_imp.shape[1]):.1%}")
    print(f"  • Data leakage              : ✓ Corregido (scaler/PCA fit solo en train)")
    print("=" * 60)
    print("\n✓ PCA completado. Datos listos para modelos.")

    return pca_optimal, X_train_pca, X_test_pca, scaler


# ── Helpers de visualización ──────────────────────────────────────────────

def _plot_variance(explained: np.ndarray, cumulative: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.bar(range(1, len(explained) + 1), explained, alpha=0.6)
    ax1.plot(range(1, len(explained) + 1), explained, "r-", marker="o")
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("Varianza Explicada")
    ax1.set_title("Varianza Explicada por Componente (train)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(cumulative) + 1), cumulative, "b-", marker="o", linewidth=2)
    ax2.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95%")
    ax2.axhline(y=0.90, color="g", linestyle="--", alpha=0.7, label="90%")
    ax2.set_xlabel("Número de Componentes")
    ax2.set_ylabel("Varianza Acumulada")
    ax2.set_title("Varianza Acumulada (train)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_2d_with_labels(X_pca_df: pd.DataFrame, y, pca: PCA,
                        title: str = "Primeras Dos Componentes Principales") -> None:
    """Visualiza PC1 vs PC2 coloreado por diagnóstico con elipses de confianza."""
    X_arr = X_pca_df.values
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr,
                          cmap="coolwarm", alpha=0.7, s=50, edgecolors="k")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
    plt.title(title)
    plt.colorbar(scatter, label="Diagnóstico (0=Benigno, 1=Maligno)")
    plt.grid(True, alpha=0.3)

    for diag_val in [0, 1]:
        mask = y_arr == diag_val
        if mask.sum() > 1:
            cov = np.cov(X_arr[mask, 0], X_arr[mask, 1])
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(np.abs(lambda_))
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
    from models import split_data

    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, data_processed = preprocess(raw)
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_imputed, y)
    pca_optimal, X_train_pca, X_test_pca, scaler = run_pca(X_train_raw, X_test_raw)
    plot_2d_with_labels(X_train_pca, y_train, pca_optimal, "PCA — Conjunto de Entrenamiento")
