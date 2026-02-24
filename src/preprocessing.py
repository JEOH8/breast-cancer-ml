"""
preprocessing.py
================
Módulo de carga y preprocesamiento del Wisconsin Breast Cancer Dataset.
"""

import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.impute import SimpleImputer


def download_dataset() -> str:
    """
    Descarga el dataset desde Kaggle usando kagglehub.

    Returns
    -------
    str
        Ruta al directorio con los archivos descargados.
    """
    file_path = kagglehub.dataset_download("imtkaggleteam/breast-cancer")
    print(f"Dataset descargado en: {file_path}")
    return file_path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga el archivo CSV del dataset.

    Parameters
    ----------
    file_path : str
        Ruta al directorio que contiene el CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos originales.
    """
    csv_path = os.path.join(file_path, "breast-cancer-wisconsin-data_data.csv")
    data = pd.read_csv(csv_path)

    print("=" * 60)
    print("INFORMACIÓN INICIAL DEL DATASET")
    print("=" * 60)
    print(f"\n  Dimensiones: {data.shape[0]} filas × {data.shape[1]} columnas")
    print(f"\n  Tipos de datos:\n{data.dtypes.value_counts()}")
    print(f"\n  Valores nulos:\n{data.isnull().sum()[data.isnull().sum() > 0]}")

    diagnosis_counts = data["diagnosis"].value_counts()
    diagnosis_pct = data["diagnosis"].value_counts(normalize=True) * 100
    print(f"\n  Distribución de diagnóstico:")
    print(pd.DataFrame({"Conteo": diagnosis_counts, "Porcentaje (%)": diagnosis_pct}))

    return data


def preprocess(data: pd.DataFrame):
    """
    Limpia y transforma el DataFrame:
      - Elimina columnas irrelevantes (id, Unnamed: 32).
      - Codifica la variable objetivo (B → 0, M → 1).
      - Imputa valores nulos con la mediana.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame original cargado desde el CSV.

    Returns
    -------
    X_imputed : pd.DataFrame
        Características limpias e imputadas.
    y : pd.Series
        Variable objetivo binaria (0 = Benigno, 1 = Maligno).
    data_processed : pd.DataFrame
        Unión de X_imputed + y, útil para EDA.
    """
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO DE DATOS")
    print("=" * 60)

    # Eliminar columnas no relevantes
    columns_to_drop = ["id", "Unnamed: 32"]
    data_clean = data.drop(columns=columns_to_drop, errors="ignore")
    print(f"\n  Columnas eliminadas: {columns_to_drop}")
    print(f"  Nuevo shape: {data_clean.shape}")

    # Separar X e y
    X = data_clean.drop("diagnosis", axis=1)
    y = data_clean["diagnosis"].map({"B": 0, "M": 1})

    print(f"\n  Características (X): {X.shape}")
    print(f"  Variable objetivo (y): distribución {dict(zip(['Benigno(0)', 'Maligno(1)'], np.bincount(y)))}")

    # Imputación de valores nulos
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        print(f"\n  Valores nulos imputados: {X.isnull().sum().sum()}")
    else:
        X_imputed = X.copy()
        print("\n  ✓ No hay valores nulos — sin necesidad de imputación.")

    data_processed = pd.concat([X_imputed, y.rename("diagnosis").reset_index(drop=True)], axis=1)

    print("\n✓ Preprocesamiento completado.")
    return X_imputed, y, data_processed


if __name__ == "__main__":
    path = download_dataset()
    raw = load_data(path)
    X_imputed, y, data_processed = preprocess(raw)
