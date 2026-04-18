"""
Funciones para cargar datos desde archivos CSV, Excel, etc.
"""

import pandas as pd

def load_csv(path):
    """Carga un archivo CSV y retorna un DataFrame."""
    return pd.read_csv(path)
