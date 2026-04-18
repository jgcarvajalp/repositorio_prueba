"""
models.py
---------
Módulo para entrenamiento de múltiples modelos de Machine Learning.
"""

import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


MODELOS_CLASIFICACION = {
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
}

MODELOS_REGRESION = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}


def entrenar_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tarea: str = "clasificacion",
    modelos: dict = None,
) -> dict:
    """
    Entrena múltiples modelos con los datos de entrenamiento.

    Parameters
    ----------
    X_train : pd.DataFrame
        Características de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.
    tarea : str
        'clasificacion' o 'regresion'.
    modelos : dict, opcional
        Diccionario personalizado {nombre: estimador}. Si es None se
        usan los modelos por defecto según la tarea.

    Returns
    -------
    dict
        Diccionario {nombre: modelo_entrenado}.
    """
    if modelos is None:
        if tarea == "clasificacion":
            modelos = MODELOS_CLASIFICACION
        elif tarea == "regresion":
            modelos = MODELOS_REGRESION
        else:
            raise ValueError("'tarea' debe ser 'clasificacion' o 'regresion'.")

    modelos_entrenados = {}
    for nombre, modelo in modelos.items():
        print(f"Entrenando: {nombre} ...", end=" ")
        modelo.fit(X_train, y_train)
        modelos_entrenados[nombre] = modelo
        print("OK")

    return modelos_entrenados


def guardar_modelo(modelo, nombre: str, directorio: str = "models") -> str:
    """
    Guarda un modelo entrenado en disco usando joblib.

    Parameters
    ----------
    modelo : estimador de scikit-learn
        Modelo entrenado.
    nombre : str
        Nombre base del archivo (sin extensión).
    directorio : str
        Carpeta donde se guardará el modelo.

    Returns
    -------
    str
        Ruta completa del archivo guardado.
    """
    os.makedirs(directorio, exist_ok=True)
    ruta = os.path.join(directorio, f"{nombre}.joblib")
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en: {ruta}")
    return ruta


def cargar_modelo(ruta: str):
    """
    Carga un modelo previamente guardado con joblib.

    Parameters
    ----------
    ruta : str
        Ruta al archivo del modelo.

    Returns
    -------
    Estimador de scikit-learn cargado.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")
    return joblib.load(ruta)
