# Proyecto de Machine Learning

Estructura base para proyectos de machine learning con Python, pandas, scikit-learn, seaborn y matplotlib.

## Estructura de carpetas

- `data/raw/`: Datos originales sin procesar
- `data/processed/`: Datos limpios y listos para modelar
- `notebooks/`: Jupyter notebooks para exploración y reportes
- `src/`: Código fuente del proyecto
	- `data/`: Carga de datos
	- `eda/`: Análisis exploratorio
	- `preprocessing/`: Preprocesamiento
	- `models/`: Entrenamiento de modelos
	- `evaluation/`: Evaluación y comparación de modelos
	- `utils.py`: Utilidades generales
- `outputs/figures/`: Gráficas generadas
- `outputs/models/`: Modelos serializados
- `tests/`: Pruebas unitarias

## Instalación

Instala las dependencias con:

```
pip install -r requirements.txt
```

## Uso

### Paso a paso para ejecutar el proyecto

1. **Copia automática de datos:**
	- Ejecuta el script para copiar el archivo de datos original al proyecto:
	  ```
	  python src/data/copy_data.py
	  ```
	- Esto copiará el archivo `diabetes_data.csv` desde la ruta de origen a `data/raw/`.

2. **Carga y análisis exploratorio:**
	- Ejecuta el análisis exploratorio básico con:
	  ```
	  python src/eda/exploratory_run.py
	  ```
	- Esto mostrará información general, estadísticas descriptivas, valores nulos, histogramas y el mapa de calor de correlación de las variables.

3. **Instalación de dependencias:**
	- Si no lo has hecho, instala las dependencias:
	  ```
	  pip install -r requirements.txt
	  ```

4. **Estructura recomendada:**
	- Coloca tus datos originales en `data/raw/` (o usa el script de copia).
	- Usa los notebooks para exploración y prototipado.
	- Implementa y ejecuta los scripts en `src/` para el flujo completo de ML.
	- Guarda resultados y modelos en `outputs/`.

---

Incluye tus instrucciones y documentación adicional aquí.
