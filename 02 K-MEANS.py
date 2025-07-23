import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Solicitar la ruta del archivo CSV
csv_path = input("Por favor, pega la ruta completa del archivo CSV: ")

# Leer el archivo CSV
try:
    data = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Verificar que las columnas de coordenadas existan
if data.shape[1] < 2:
    print("El archivo CSV debe tener al menos 2 columnas (X, Y).")
    exit()

# Solicitar las columnas de elementos siderófilos y calcófilos
print("\nColumnas disponibles en el CSV:", list(data.columns))
siderophile_cols = input("Ingresa los nombres de las columnas de elementos siderófilos (separados por comas): ").split(',')
calcophile_cols = input("Ingresa los nombres de las columnas de elementos litófilos (separados por comas): ").split(',')

# Limpiar espacios en los nombres de las columnas
siderophile_cols = [col.strip() for col in siderophile_cols]
calcophile_cols = [col.strip() for col in calcophile_cols]

# Verificar que las columnas existan
all_cols = siderophile_cols + calcophile_cols
if not all(col in data.columns for col in all_cols):
    print("Una o más columnas especificadas no existen en el CSV.")
    exit()

# Solicitar el número de clústeres
n_clusters = int(input("Ingresa el número de clústeres para K-Means: "))

# Preparar los datos para el clustering
X = data[all_cols].dropna()

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled) + 1  # Sumar 1 para que los clústeres comiencen en 1

# Crear DataFrame con coordenadas y clústeres para exportar
export_data = data[[data.columns[0], data.columns[1], 'Cluster']].copy()
export_data.columns = ['X', 'Y', 'Dominio_Geoquimico']

# Exportar a CSV en la misma carpeta
output_csv = os.path.join(os.path.dirname(csv_path), 'geochemical_domains.csv')
export_data.to_csv(output_csv, index=False)

print(f"Datos de dominios geoquímicos exportados como: {output_csv}")