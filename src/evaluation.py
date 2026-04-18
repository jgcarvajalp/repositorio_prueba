"""
evaluation.py
-------------
Módulo para evaluación de modelos de Machine Learning.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------

def evaluar_clasificacion(
    modelos_entrenados: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evalúa múltiples modelos de clasificación y retorna un resumen de métricas.

    Parameters
    ----------
    modelos_entrenados : dict
        Diccionario {nombre: modelo_entrenado}.
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    pd.DataFrame
        Tabla con métricas por modelo, ordenada por F1-score descendente.
    """
    resultados = []
    for nombre, modelo in modelos_entrenados.items():
        y_pred = modelo.predict(X_test)
        promedio = "binary" if y_test.nunique() == 2 else "weighted"
        fila = {
            "Modelo": nombre,
            "Exactitud": accuracy_score(y_test, y_pred),
            "Precisión": precision_score(y_test, y_pred, average=promedio, zero_division=0),
            "Recall": recall_score(y_test, y_pred, average=promedio, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average=promedio, zero_division=0),
        }
        # ROC-AUC solo si el modelo soporta predict_proba
        if hasattr(modelo, "predict_proba"):
            try:
                y_proba = modelo.predict_proba(X_test)
                if y_test.nunique() == 2:
                    fila["ROC-AUC"] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    fila["ROC-AUC"] = roc_auc_score(
                        y_test, y_proba, multi_class="ovr", average="weighted"
                    )
            except ValueError as exc:
                print(f"  [Advertencia] No se pudo calcular ROC-AUC para '{nombre}': {exc}")
                fila["ROC-AUC"] = None
        resultados.append(fila)

    df_resultados = pd.DataFrame(resultados).sort_values("F1-Score", ascending=False)
    df_resultados.reset_index(drop=True, inplace=True)
    return df_resultados


def graficar_matriz_confusion(
    modelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    nombre_modelo: str = "Modelo",
    directorio: str = "reports/figures",
) -> None:
    """
    Grafica y guarda la matriz de confusión de un modelo de clasificación.

    Parameters
    ----------
    modelo : estimador de scikit-learn
    X_test : pd.DataFrame
    y_test : pd.Series
    nombre_modelo : str
    directorio : str
    """
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    etiquetas = sorted(y_test.unique())

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas,
        yticklabels=etiquetas,
    )
    plt.title(f"Matriz de Confusión — {nombre_modelo}")
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predicho")
    plt.tight_layout()

    os.makedirs(directorio, exist_ok=True)
    nombre_archivo = nombre_modelo.replace(" ", "_").lower()
    ruta = os.path.join(directorio, f"confusion_matrix_{nombre_archivo}.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Matriz de confusión guardada en: {ruta}")


# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------

def evaluar_regresion(
    modelos_entrenados: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evalúa múltiples modelos de regresión y retorna un resumen de métricas.

    Parameters
    ----------
    modelos_entrenados : dict
        Diccionario {nombre: modelo_entrenado}.
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    pd.DataFrame
        Tabla con métricas por modelo, ordenada por R² descendente.
    """
    resultados = []
    for nombre, modelo in modelos_entrenados.items():
        y_pred = modelo.predict(X_test)
        resultados.append({
            "Modelo": nombre,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
        })

    df_resultados = pd.DataFrame(resultados).sort_values("R²", ascending=False)
    df_resultados.reset_index(drop=True, inplace=True)
    return df_resultados


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def graficar_comparacion_modelos(
    df_resultados: pd.DataFrame,
    metrica: str,
    titulo: str = "Comparación de Modelos",
    directorio: str = "reports/figures",
) -> None:
    """
    Genera un gráfico de barras comparando modelos según una métrica.

    Parameters
    ----------
    df_resultados : pd.DataFrame
        DataFrame con columnas 'Modelo' y la métrica a graficar.
    metrica : str
        Nombre de la columna de la métrica (ej. 'F1-Score', 'R²').
    titulo : str
    directorio : str
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resultados, x=metrica, y="Modelo", palette="viridis")
    plt.title(titulo)
    plt.xlabel(metrica)
    plt.ylabel("Modelo")
    plt.tight_layout()

    os.makedirs(directorio, exist_ok=True)
    nombre_archivo = metrica.replace(" ", "_").replace("²", "2").lower()
    ruta = os.path.join(directorio, f"comparacion_{nombre_archivo}.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Gráfico de comparación guardado en: {ruta}")
