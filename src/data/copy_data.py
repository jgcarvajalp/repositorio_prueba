import os
import shutil

# Ruta de origen y destino
src = r"C:\Users\cpj33\Documents\UNIVERSIDAD_BOSQUE\INTRODUCCION_MACHINE_LEARNING\datos\diabetes_data.csv"
dst = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'diabetes_data.csv')
dst = os.path.abspath(dst)

if not os.path.exists(src):
    print(f"El archivo de origen no existe: {src}")
else:
    shutil.copy2(src, dst)
    print(f"Archivo copiado a: {dst}")
