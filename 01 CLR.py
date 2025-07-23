import pandas as pd
import numpy as np
from pathlib import Path

# Solicitar la ruta del archivo CSV
csv_path = input("Por favor, ingrese la ruta completa del archivo CSV: ")

# Leer el archivo CSV
df = pd.read_csv(csv_path)

# Seleccionar solo las columnas numéricas (concentraciones de elementos)
numeric_cols = df.select_dtypes(include=[np.number]).columns
data_numeric = df[numeric_cols]

# Reemplazar ceros con un valor pequeño positivo (por ejemplo, 0.0001)
# Esto es una práctica común en geoquímica para manejar ceros en CLR
small_value = 0.0001
data_numeric = data_numeric.where(data_numeric > 0, small_value)

# Verificar que no haya valores no positivos después del reemplazo
if (data_numeric <= 0).any().any():
    raise ValueError("Los datos aún contienen valores no positivos después del reemplazo.")

# Calcular la media geométrica por fila
def geometric_mean(row):
    return np.exp(np.mean(np.log(row)))

# Calcular CLR para cada elemento
clr_data = data_numeric.apply(lambda x: np.log(x / geometric_mean(x)), axis=1)

# Crear un nuevo DataFrame con los datos transformados
result_df = df.copy()
result_df[numeric_cols] = clr_data

# Crear la ruta de salida (mismo directorio, nombre modificado)
output_path = Path(csv_path).parent / f"{Path(csv_path).stem}_clr_normalized.csv"

# Guardar el resultado en un nuevo CSV
result_df.to_csv(output_path, index=False)
print(f"Archivo normalizado guardado en: {output_path}")