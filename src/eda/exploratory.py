"""
Funciones para análisis exploratorio y visualización de datos.
"""

import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(df, column):
    sns.histplot(df[column])
    plt.show()
