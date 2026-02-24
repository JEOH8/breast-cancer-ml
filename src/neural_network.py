"""
neural_network.py
=================
Red Neuronal Profunda con TensorFlow/Keras para clasificación de cáncer de mama.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ── Arquitectura ────────────────────────────────────────────────────────────

def build_neural_network(input_dim: int,
                         dropout_rate: float = 0.3,
                         l2_reg: float = 0.001) -> keras.Sequential:
    """
    Construye y compila la red neuronal:
      Entrada → Dense(64)+BN+Drop → Dense(32)+BN+Drop → Dense(16)+BN+Drop → Sigmoid(1)

    Parameters
    ----------
    input_dim : int
    dropout_rate : float
    l2_reg : float

    Returns
    -------
    model : keras.Sequential compilado
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.8),

        layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg * 0.5),
                     kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.6),

        layers.Dense(1, activation="sigmoid",
                     kernel_initializer="glorot_uniform"),
    ])

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


# ── Entrenamiento ────────────────────────────────────────────────────────────

def train_neural_network(X_train, X_test, y_train, y_test,
                         epochs: int = 100, batch_size: int = 16):
    """
    Entrena la red neuronal con callbacks y pesos de clase balanceados.

    Returns
    -------
    best_model : keras.Model
    history : keras History object
    best_threshold : float
    """
    print("=" * 70)
    print("RED NEURONAL PROFUNDA PARA CLASIFICACIÓN")
    print("=" * 70)

    # Asegurar arrays numpy
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_te = X_test.values if hasattr(X_test, "values") else X_test

    # Pesos de clases
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\n  Pesos de clases: Benigno={class_weight_dict[0]:.3f}, Maligno={class_weight_dict[1]:.3f}")

    # Verificar GPU
    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    print(f"  TensorFlow {tf.__version__} | GPU: {'Sí' if gpu_available else 'No'}")

    # Construir modelo
    input_dim = X_tr.shape[1]
    model = build_neural_network(input_dim)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint("best_neural_model.h5", monitor="val_auc",
                        save_best_only=True, mode="max", verbose=1),
    ]

    print("\n  Iniciando entrenamiento...")
    history = model.fit(
        X_tr, y_train,
        validation_data=(X_te, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Cargar mejor modelo
    best_model = keras.models.load_model("best_neural_model.h5")

    # Evaluar
    test_results = best_model.evaluate(X_te, y_test, verbose=0)
    test_predictions = best_model.predict(X_te)
    test_pred_binary = (test_predictions > 0.5).astype(int)

    print(f"\n  Resultados (umbral 0.5):")
    print(f"    Loss      : {test_results[0]:.4f}")
    print(f"    Accuracy  : {test_results[1]:.4f}")
    print(f"    Precision : {test_results[2]:.4f}")
    print(f"    Recall    : {test_results[3]:.4f}")
    print(f"    AUC       : {test_results[4]:.4f}")

    # Optimizar umbral
    best_threshold, best_f1 = _optimize_threshold(test_predictions, y_test)

    # Visualizaciones
    _plot_training_history(history)
    _plot_evaluation(best_model, X_te, y_test, test_predictions)

    return best_model, history, best_threshold


# ── Optimización de umbral ──────────────────────────────────────────────────

def _optimize_threshold(predictions: np.ndarray, y_true):
    """Encuentra el umbral que maximiza F1-Score."""
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = [f1_score(y_true, (predictions > t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, "b-", linewidth=2)
    plt.axvline(x=best_thr, color="r", linestyle="--", label=f"Óptimo: {best_thr:.3f}")
    plt.axvline(x=0.5, color="g", linestyle="--", label="Default: 0.5")
    plt.xlabel("Umbral de Decisión")
    plt.ylabel("F1-Score")
    plt.title("Optimización del Umbral de Decisión")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\n  Umbral óptimo : {best_thr:.3f} | F1 = {best_f1:.4f}")
    return best_thr, best_f1


# ── Función de predicción ───────────────────────────────────────────────────

def predict_with_neural_network(features, model, threshold: float = None) -> dict:
    """
    Predicción interpretable con la red neuronal.

    Parameters
    ----------
    features : array-like
        Vector de componentes PCA.
    model : keras.Model
    threshold : float | None
        Si None, usa 0.5.

    Returns
    -------
    dict con predicción, probabilidades, confianza y recomendación.
    """
    features = np.array(features).reshape(1, -1)
    threshold = threshold or 0.5

    proba = float(model.predict(features, verbose=0)[0][0])
    label = "Maligno" if proba >= threshold else "Benigno"
    confidence = proba if label == "Maligno" else 1 - proba

    if confidence >= 0.9:
        level = "MUY ALTA"
    elif confidence >= 0.8:
        level = "ALTA"
    elif confidence >= 0.7:
        level = "MODERADA"
    else:
        level = "BAJA"

    return {
        "prediccion": label,
        "probabilidad_maligno": proba,
        "probabilidad_benigno": 1 - proba,
        "confianza": confidence,
        "nivel_confianza": level,
        "umbral_utilizado": threshold,
        "recomendacion": (
            "CONSULTAR CON ONCÓLOGO URGENTE"
            if label == "Maligno"
            else "SEGUIMIENTO RUTINARIO"
        ),
        "modelo_utilizado": "Red Neuronal Profunda",
    }


# ── Guardado del modelo ─────────────────────────────────────────────────────

def save_model(model, history, best_threshold: float,
               X_train, pca_optimal, opt_metrics: dict) -> None:
    """Guarda el modelo Keras y sus metadatos JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/breast_cancer_neural_model_{ts}.h5"
    meta_path = f"models/neural_model_metadata_{ts}.json"

    model.save(model_path)

    test_preds = model.predict(
        X_train.values if hasattr(X_train, "values") else X_train, verbose=0
    )

    metadata = {
        "model_info": {
            "model_type": "NeuralNetwork",
            "architecture": "64→32→16→1 (ReLU+BN+Dropout, Sigmoid output)",
            "optimizer": "Adam (lr=0.001)",
            "creation_date": ts,
        },
        "training": {
            "epochs_trained": len(history.history["loss"]),
            "batch_size": 16,
            "optimal_threshold": float(best_threshold),
        },
        "data": {
            "pca_components": X_train.shape[1],
            "variance_explained": float(pca_optimal.explained_variance_ratio_.sum()),
        },
        "performance_vs_logistic": opt_metrics,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"  ✓ Modelo guardado   : {model_path}")
    print(f"  ✓ Metadatos guardados: {meta_path}")


# ── Helpers de visualización ────────────────────────────────────────────────

def _plot_training_history(history) -> None:
    metrics = ["loss", "accuracy", "precision", "recall", "auc"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        axes[idx].plot(history.history[metric], label="Entrenamiento", lw=2)
        axes[idx].plot(history.history[f"val_{metric}"], label="Validación", lw=2)
        axes[idx].set_xlabel("Época")
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_title(f"{metric.capitalize()} durante el Entrenamiento")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    # Learning rate
    if "lr" in history.history:
        axes[5].plot(history.history["lr"], lw=2, color="purple")
        axes[5].set_yscale("log")
        axes[5].set_title("Learning Rate")
        axes[5].grid(True, alpha=0.3)
    else:
        axes[5].axis("off")

    plt.tight_layout()
    plt.show()


def _plot_evaluation(model, X_test, y_test, predictions) -> None:
    pred_binary = (predictions > 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_binary)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc_val = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Matriz de Confusión — Red Neuronal")
    axes[0].set_xticklabels(["Benigno", "Maligno"])
    axes[0].set_yticklabels(["Benigno", "Maligno"])

    axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC (AUC = {roc_auc_val:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=2, label="Aleatorio")
    axes[1].set_title("Curva ROC — Red Neuronal")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(predictions[y_test == 0], alpha=0.7, bins=20,
                 label="Benigno (Real)", color="green")
    axes[2].hist(predictions[y_test == 1], alpha=0.7, bins=20,
                 label="Maligno (Real)", color="red")
    axes[2].axvline(x=0.5, color="black", linestyle="--", label="Umbral 0.5")
    axes[2].set_title("Distribución de Probabilidades Predichas")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, pred_binary,
                                target_names=["Benigno", "Maligno"]))


if __name__ == "__main__":
    from preprocessing import download_dataset, load_data, preprocess
    from pca_analysis import run_pca
    from models import split_data

    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, _ = preprocess(raw)
    pca_optimal, X_pca_df, _ = run_pca(X_imputed)
    X_train, X_test, y_train, y_test = split_data(X_pca_df, y)

    best_model, history, best_threshold = train_neural_network(
        X_train, X_test, y_train, y_test
    )
