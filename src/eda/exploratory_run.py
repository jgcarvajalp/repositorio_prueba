"""
Script para cargar los datos y realizar análisis exploratorio básico.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    # Ruta al archivo de datos
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'diabetes_data.csv')
    data_path = os.path.abspath(data_path)

    # Cargar datos
    df = pd.read_csv(data_path)
    print("\n--- Información general ---")
    print(df.info())
    print("\n--- Primeras filas ---")
    print(df.head())
    print("\n--- Estadísticas descriptivas ---")
    print(df.describe())
    print("\n--- Valores nulos por columna ---")
    print(df.isnull().sum())

    # Histogramas de todas las variables numéricas
    print("\n--- Histogramas de variables numéricas ---")
    df.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Mapa de calor de correlación
    print("\n--- Mapa de calor de correlación ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de correlación')
    plt.show()

if __name__ == "__main__":
    main()
