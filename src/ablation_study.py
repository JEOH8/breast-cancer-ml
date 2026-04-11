# -*- coding: utf-8 -*-
"""
ablation_study.py
=================
Ablation study de arquitecturas de red neuronal para clasificación
de cáncer de mama.

Compara 4 arquitecturas entrenadas bajo las mismas condiciones
(mismo scaler, PCA, split, semillas, callbacks) para determinar
cuánto aporta cada capa a la capacidad predictiva del modelo.

Arquitecturas evaluadas:
  A — 64→32→16→1  (baseline actual)
  B — 32→16→1     (más simple, menos parámetros)
  C — 128→64→32→1 (más grande, mayor capacidad)
  D — 64→1        (una sola capa oculta)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


# ── Definición de arquitecturas ────────────────────────────────────────────

ARCHITECTURES = {
    "A — 64→32→16→1\n(Baseline)": [64, 32, 16],
    "B — 32→16→1\n(Simple)":      [32, 16],
    "C — 128→64→32→1\n(Grande)":  [128, 64, 32],
    "D — 64→1\n(1 capa)":         [64],
}


def set_seeds(seed: int = 42) -> None:
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_architecture(layer_sizes: list,
                        input_dim: int,
                        dropout_rate: float = 0.3,
                        l2_reg: float = 0.001) -> keras.Sequential:
    """
    Construye una red neuronal con la arquitectura especificada.
    Todas las capas usan ReLU + BatchNorm + Dropout para comparación justa.

    Parameters
    ----------
    layer_sizes : list de int — neuronas por capa oculta
    input_dim   : int
    dropout_rate: float
    l2_reg      : float
    """
    model_layers = [layers.Input(shape=(input_dim,))]

    for i, size in enumerate(layer_sizes):
        # Dropout decrece progresivamente en capas más profundas
        drop = dropout_rate * (1 - i * 0.1)
        reg  = l2_reg * (0.5 ** i)

        model_layers += [
            layers.Dense(size, activation="relu",
                         kernel_regularizer=regularizers.l2(reg),
                         kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(max(drop, 0.1)),
        ]

    model_layers.append(
        layers.Dense(1, activation="sigmoid",
                     kernel_initializer="glorot_uniform")
    )

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def count_params(layer_sizes: list, input_dim: int) -> int:
    """Cuenta parámetros entrenables de una arquitectura."""
    model = build_architecture(layer_sizes, input_dim)
    return model.count_params()


def train_architecture(name: str,
                        layer_sizes: list,
                        X_train, X_test,
                        y_train, y_test,
                        epochs: int = 100,
                        batch_size: int = 16) -> dict:
    """
    Entrena una arquitectura y devuelve métricas completas.

    Parameters
    ----------
    name        : str — nombre descriptivo de la arquitectura
    layer_sizes : list — neuronas por capa oculta
    X_train, X_test : arrays numpy
    y_train, y_test : arrays
    epochs      : int
    batch_size  : int

    Returns
    -------
    dict con métricas, historia y parámetros del modelo
    """
    set_seeds(42)

    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_te = X_test.values  if hasattr(X_test,  "values") else X_test

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model = build_architecture(layer_sizes, X_tr.shape[1])
    n_params = model.count_params()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_tr, y_train,
        validation_data=(X_te, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0,
    )

    # Métricas con umbral 0.5
    preds_raw = model.predict(X_te, verbose=0).flatten()
    preds_bin = (preds_raw > 0.5).astype(int)

    # Optimización del umbral
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores  = [f1_score(y_test, (preds_raw > t).astype(int))
                  for t in thresholds]
    best_thr   = thresholds[np.argmax(f1_scores)]
    preds_opt  = (preds_raw > best_thr).astype(int)

    epochs_trained = len(history.history["loss"])
    best_val_epoch = np.argmax(history.history["val_auc"])

    return {
        "Arquitectura":   name.replace("\n", " "),
        "Capas":          str(layer_sizes) + "→[1]",
        "Parámetros":     n_params,
        "Épocas":         epochs_trained,
        "Accuracy":       accuracy_score(y_test, preds_opt),
        "Precision":      precision_score(y_test, preds_opt),
        "Recall":         recall_score(y_test, preds_opt),
        "F1-Score":       f1_score(y_test, preds_opt),
        "AUC":            roc_auc_score(y_test, preds_raw),
        "Umbral óptimo":  best_thr,
        "Val AUC (mejor)": history.history["val_auc"][best_val_epoch],
        "_history":       history,
        "_model":         model,
    }


# ── Ablation study completo ────────────────────────────────────────────────

def run_ablation_study(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Ejecuta el ablation study completo sobre las 4 arquitecturas.

    Returns
    -------
    results_df : pd.DataFrame con métricas comparativas
    histories  : dict de historiales de entrenamiento
    """
    print("=" * 70)
    print("ABLATION STUDY — ARQUITECTURAS DE RED NEURONAL")
    print("(Mismas condiciones: semillas, callbacks, class_weight, epochs=100)")
    print("=" * 70)

    results  = []
    histories = {}

    for name, layer_sizes in ARCHITECTURES.items():
        clean_name = name.replace("\n", " ")
        n_params = count_params(
            layer_sizes,
            X_train.shape[1] if hasattr(X_train, "shape") else len(X_train.columns)
        )
        print(f"\n  Entrenando {clean_name} | {n_params:,} parámetros...")

        result = train_architecture(
            name, layer_sizes,
            X_train, X_test, y_train, y_test
        )
        results.append(result)
        histories[clean_name] = result["_history"]

        print(f"    Épocas: {result['Épocas']:3d} | "
              f"Accuracy: {result['Accuracy']:.4f} | "
              f"F1: {result['F1-Score']:.4f} | "
              f"AUC: {result['AUC']:.4f}")

    results_df = pd.DataFrame(results).drop(columns=["_history", "_model"])
    results_df = results_df.sort_values("F1-Score", ascending=False)

    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA — ABLATION STUDY")
    print("=" * 70)
    display_cols = ["Arquitectura", "Parámetros", "Épocas",
                    "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    print(results_df[display_cols].to_string(index=False))

    return results_df, histories


# ── Visualizaciones ────────────────────────────────────────────────────────

def plot_ablation_results(results_df: pd.DataFrame,
                           histories: dict) -> None:
    """
    Genera 4 gráficos del ablation study:
      1. Comparativa de métricas por arquitectura
      2. Parámetros vs F1-Score
      3. Curvas de loss durante entrenamiento
      4. Curvas de AUC durante entrenamiento
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Ablation Study — Arquitecturas de Red Neuronal\n"
                 "Wisconsin Breast Cancer Dataset",
                 fontsize=16, fontweight="bold", y=0.98)

    nombres = results_df["Arquitectura"].tolist()
    x = np.arange(len(nombres))
    width = 0.2
    colors_metrics = ["steelblue", "mediumseagreen", "lightcoral", "darkorchid"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # ── 1. Comparativa de métricas ────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
        vals = results_df.set_index("Arquitectura").loc[nombres, metric].values
        bars = ax1.bar(x + i * width, vals, width,
                       label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold")

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(nombres, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0.85, 1.04)
    ax1.set_ylabel("Valor")
    ax1.set_title("Comparativa de Métricas por Arquitectura")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Línea de referencia: F1 del modelo base
    baseline_f1 = results_df[results_df["Arquitectura"].str.contains("Baseline")]["F1-Score"].values
    if len(baseline_f1) > 0:
        ax1.axhline(y=baseline_f1[0], color="red", linestyle="--",
                    alpha=0.5, label=f"Baseline F1={baseline_f1[0]:.3f}")

    # ── 2. Parámetros vs F1-Score ─────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    params = results_df["Parámetros"].values
    f1s    = results_df["F1-Score"].values
    aucs   = results_df["AUC"].values
    arqs   = results_df["Arquitectura"].values

    scatter = ax2.scatter(params, f1s, s=200, c=aucs,
                          cmap="RdYlGn", vmin=0.99, vmax=1.0,
                          zorder=5, edgecolors="black", linewidth=1)
    plt.colorbar(scatter, ax=ax2, label="AUC")

    for arq, p, f in zip(arqs, params, f1s):
        label = arq.split("(")[0].strip()
        ax2.annotate(label, xy=(p, f),
                     xytext=(8, 4), textcoords="offset points",
                     fontsize=9, fontweight="bold")

    ax2.set_xlabel("Número de Parámetros Entrenables")
    ax2.set_ylabel("F1-Score")
    ax2.set_title("Complejidad del Modelo vs Rendimiento\n"
                  "Color = AUC (verde = mejor)")
    ax2.grid(True, alpha=0.3)

    # ── 3. Curvas de loss ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    colors_arch = ["royalblue", "forestgreen", "crimson", "darkorchid"]
    for (name, hist), color in zip(histories.items(), colors_arch):
        label = name.split("(")[0].strip()
        epochs_range = range(1, len(hist.history["loss"]) + 1)
        ax3.plot(epochs_range, hist.history["loss"],
                 color=color, lw=2, label=f"{label} — train", alpha=0.8)
        ax3.plot(epochs_range, hist.history["val_loss"],
                 color=color, lw=2, linestyle="--",
                 label=f"{label} — val", alpha=0.5)

    ax3.set_xlabel("Época")
    ax3.set_ylabel("Loss (Binary Crossentropy)")
    ax3.set_title("Curvas de Loss durante Entrenamiento\n"
                  "Sólido = train | Discontinuo = validación")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── 4. Curvas de AUC de validación ────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    for (name, hist), color in zip(histories.items(), colors_arch):
        label = name.split("(")[0].strip()
        epochs_range = range(1, len(hist.history["val_auc"]) + 1)
        ax4.plot(epochs_range, hist.history["val_auc"],
                 color=color, lw=2, label=label)

    ax4.set_xlabel("Época")
    ax4.set_ylabel("AUC — Conjunto de Validación")
    ax4.set_title("Evolución del AUC de Validación\n"
                  "Convergencia y estabilidad por arquitectura")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.95, 1.005)

    plt.tight_layout()
    plt.show()

    # Conclusiones automáticas
    print("\n" + "=" * 70)
    print("CONCLUSIONES DEL ABLATION STUDY")
    print("=" * 70)

    best  = results_df.iloc[0]
    worst = results_df.iloc[-1]
    base  = results_df[results_df["Arquitectura"].str.contains("Baseline")]

    print(f"\n  Mejor arquitectura  : {best['Arquitectura']}")
    print(f"    F1-Score          : {best['F1-Score']:.4f}")
    print(f"    AUC               : {best['AUC']:.4f}")
    print(f"    Parámetros        : {best['Parámetros']:,}")

    print(f"\n  Peor arquitectura   : {worst['Arquitectura']}")
    print(f"    F1-Score          : {worst['F1-Score']:.4f}")
    print(f"    Degradación vs mejor: {best['F1-Score'] - worst['F1-Score']:+.4f}")

    if len(base) > 0:
        b = base.iloc[0]
        print(f"\n  Baseline (64→32→16→1):")
        print(f"    F1-Score          : {b['F1-Score']:.4f}")
        print(f"    Posición en ranking: "
              f"{results_df[results_df['Arquitectura'] == b['Arquitectura']].index[0] + 1}"
              f"/{len(results_df)}")

    diff = results_df["F1-Score"].max() - results_df["F1-Score"].min()
    print(f"\n  Rango de F1 entre arquitecturas: {diff:.4f} ({diff*100:.2f}%)")
    if diff < 0.02:
        print("  → La arquitectura tiene impacto MÍNIMO en este dataset.")
        print("    El problema es suficientemente simple para cualquier configuración.")
    elif diff < 0.05:
        print("  → La arquitectura tiene impacto MODERADO.")
        print("    Vale la pena seleccionar cuidadosamente las capas.")
    else:
        print("  → La arquitectura tiene impacto SIGNIFICATIVO.")
        print("    La selección de capas es crítica para el rendimiento.")


# ── Tabla resumen para README/LinkedIn ────────────────────────────────────

def print_summary_table(results_df: pd.DataFrame) -> None:
    """Imprime tabla de resumen en formato Markdown."""
    print("\n### Tabla Markdown para README:\n")
    print("| Arquitectura | Parámetros | Accuracy | Precision | Recall | F1-Score | AUC |")
    print("|---|---|---|---|---|---|---|")
    for _, row in results_df.iterrows():
        arch = row["Arquitectura"].replace("(Baseline)", "⭐ (Baseline)")
        print(f"| {arch} | {row['Parámetros']:,} | {row['Accuracy']:.4f} | "
              f"{row['Precision']:.4f} | {row['Recall']:.4f} | "
              f"{row['F1-Score']:.4f} | {row['AUC']:.4f} |")


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess
    from models import split_data
    from pca_analysis import run_pca
    import sys
    sys.path.insert(0, "src")

    path = download_dataset()
    raw  = load_data(path)
    X_imputed, y, _ = preprocess(raw)
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_imputed, y)
    _, X_train_pca, X_test_pca, _ = run_pca(X_train_raw, X_test_raw)

    results_df, histories = run_ablation_study(
        X_train_pca, X_test_pca, y_train, y_test
    )
    plot_ablation_results(results_df, histories)
    print_summary_table(results_df)
