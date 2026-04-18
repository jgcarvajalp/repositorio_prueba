"""
data_loader.py
--------------
Módulo para carga y validación del conjunto de datos.
"""

import os
import pandas as pd


def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga un conjunto de datos desde un archivo CSV, Excel, JSON o Parquet.

    Parameters
    ----------
    ruta : str
        Ruta al archivo de datos.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si la extensión del archivo no es compatible.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    extension = os.path.splitext(ruta)[1].lower()

    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
    }

    if extension not in loaders:
        raise ValueError(
            f"Extensión '{extension}' no soportada. "
            f"Extensiones válidas: {list(loaders.keys())}"
        )

    df = loaders[extension](ruta)
    print(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def resumen_datos(df: pd.DataFrame) -> None:
    """
    Imprime un resumen básico del DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Conjunto de datos a describir.
    """
    print("=" * 50)
    print("RESUMEN DEL CONJUNTO DE DATOS")
    print("=" * 50)
    print(f"Dimensiones : {df.shape}")
    print(f"Columnas    : {list(df.columns)}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("\nEstadísticas descriptivas:")
    print(df.describe(include="all"))
