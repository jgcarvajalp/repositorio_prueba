"""
preprocessing.py
----------------
Módulo para preprocesamiento del conjunto de datos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def separar_caracteristicas_objetivo(
    df: pd.DataFrame, columna_objetivo: str
) -> tuple:
    """
    Separa las características (X) de la variable objetivo (y).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo.
    columna_objetivo : str
        Nombre de la columna objetivo.

    Returns
    -------
    tuple
        (X, y) donde X es un DataFrame y y es una Serie.
    """
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]
    return X, y


def dividir_conjunto(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento, (validación) y prueba.

    Parameters
    ----------
    X : pd.DataFrame
        Características.
    y : pd.Series
        Variable objetivo.
    test_size : float
        Proporción del conjunto de prueba.
    val_size : float
        Proporción del conjunto de validación (0 para omitirlo).
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    tuple
        Sin validación: (X_train, X_test, y_train, y_test)
        Con validación: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Apply stratified splitting for classification tasks (few unique target values)
    _MAX_CLASSES_STRATIFY = 20
    stratify = y if y.nunique() <= _MAX_CLASSES_STRATIFY else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    if val_size > 0:
        val_relative = val_size / (1 - test_size)
        stratify_val = y_train if y_train.nunique() <= _MAX_CLASSES_STRATIFY else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_relative, random_state=random_state,
            stratify=stratify_val
        )
        print(
            f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def imputar_valores_faltantes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    estrategia_numerica: str = "mean",
    estrategia_categorica: str = "most_frequent",
) -> tuple:
    """
    Imputa valores faltantes en columnas numéricas y categóricas.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    estrategia_numerica : str
        Estrategia para columnas numéricas ('mean', 'median', 'most_frequent').
    estrategia_categorica : str
        Estrategia para columnas categóricas ('most_frequent', 'constant').

    Returns
    -------
    tuple
        (X_train_imputado, X_test_imputado)
    """
    numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train_out = X_train.copy()
    X_test_out = X_test.copy()

    if numericas:
        imputer_num = SimpleImputer(strategy=estrategia_numerica)
        X_train_out[numericas] = imputer_num.fit_transform(X_train[numericas])
        X_test_out[numericas] = imputer_num.transform(X_test[numericas])

    if categoricas:
        imputer_cat = SimpleImputer(strategy=estrategia_categorica)
        X_train_out[categoricas] = imputer_cat.fit_transform(X_train[categoricas])
        X_test_out[categoricas] = imputer_cat.transform(X_test[categoricas])

    return X_train_out, X_test_out


def escalar_caracteristicas(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple:
    """
    Escala las características numéricas con StandardScaler.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple
        (X_train_escalado, X_test_escalado, scaler)
    """
    numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()

    scaler = StandardScaler()
    X_train_out[numericas] = scaler.fit_transform(X_train[numericas])
    X_test_out[numericas] = scaler.transform(X_test[numericas])

    return X_train_out, X_test_out, scaler


def codificar_variables_categoricas(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple:
    """
    Codifica variables categóricas con One-Hot Encoding.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple
        (X_train_codificado, X_test_codificado)
    """
    categoricas = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    if not categoricas:
        return X_train, X_test

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_train = encoder.fit_transform(X_train[categoricas])
    encoded_test = encoder.transform(X_test[categoricas])

    feature_names = encoder.get_feature_names_out(categoricas)
    df_train_enc = pd.DataFrame(encoded_train, columns=feature_names, index=X_train.index)
    df_test_enc = pd.DataFrame(encoded_test, columns=feature_names, index=X_test.index)

    X_train_out = pd.concat(
        [X_train.drop(columns=categoricas), df_train_enc], axis=1, ignore_index=False
    ).reset_index(drop=True)
    X_test_out = pd.concat(
        [X_test.drop(columns=categoricas), df_test_enc], axis=1, ignore_index=False
    ).reset_index(drop=True)

    return X_train_out, X_test_out
