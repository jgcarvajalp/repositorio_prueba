# Proyecto de Machine Learning

Repositorio para desarrollar un proyecto completo de aprendizaje de máquina: carga y exploración del conjunto de datos, preprocesamiento, entrenamiento y evaluación de múltiples modelos.

---

## 📁 Estructura del proyecto

```
repositorio_prueba/
├── data/
│   ├── raw/                  # Datos originales (sin modificar)
│   └── processed/            # Datos listos para modelado
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb   # EDA
│   ├── 02_preprocesamiento.ipynb        # Limpieza y transformaciones
│   ├── 03_entrenamiento_modelos.ipynb   # Entrenamiento de modelos
│   └── 04_evaluacion_modelos.ipynb      # Evaluación y comparación
├── src/
│   ├── data_loader.py        # Carga y resumen de datos
│   ├── preprocessing.py      # Transformaciones del dataset
│   ├── models.py             # Entrenamiento y persistencia de modelos
│   └── evaluation.py         # Métricas y visualizaciones de evaluación
├── models/                   # Modelos entrenados (.joblib)
├── reports/
│   └── figures/              # Gráficas generadas
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/jgcarvajalp/repositorio_prueba.git
cd repositorio_prueba

# 2. (Opcional) Crea un entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Instala las dependencias
pip install -r requirements.txt
```

---

## 🚀 Flujo de trabajo

Sigue los notebooks en orden:

| Notebook | Descripción |
|---|---|
| `01_analisis_exploratorio.ipynb` | Carga del dataset, estadísticas, visualizaciones |
| `02_preprocesamiento.ipynb` | Limpieza, imputación, codificación y escalado |
| `03_entrenamiento_modelos.ipynb` | Entrenamiento de múltiples algoritmos |
| `04_evaluacion_modelos.ipynb` | Métricas, matrices de confusión y comparación |

### Usar tu propio dataset

1. Coloca el archivo de datos en `data/raw/` (formatos soportados: `.csv`, `.xlsx`, `.json`, `.parquet`).
2. En cada notebook, actualiza la variable `columna_objetivo` y la ruta al archivo.

---

## 🤖 Modelos disponibles

### Clasificación
- Regresión Logística
- Árbol de Decisión
- Random Forest
- Gradient Boosting
- SVM
- K-Nearest Neighbors (KNN)

### Regresión
- Regresión Lineal
- Árbol de Decisión
- Random Forest
- Gradient Boosting
- SVR
- K-Nearest Neighbors (KNN)

---

## 📊 Métricas de evaluación

| Tarea | Métricas |
|---|---|
| Clasificación | Exactitud, Precisión, Recall, F1-Score, ROC-AUC |
| Regresión | RMSE, MAE, R² |
