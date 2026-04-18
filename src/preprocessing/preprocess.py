"""
Funciones para limpieza y transformación de datos.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
