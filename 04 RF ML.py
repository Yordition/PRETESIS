import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Solicitar al usuario la ruta del archivo CSV
ruta_csv = input("Por favor, ingrese la ruta completa del archivo CSV con los datos geoquímicos: ")

# Leer el archivo CSV
try:
    datos = pd.read_csv(ruta_csv)
except FileNotFoundError:
    print("Error: El archivo no se encuentra en la ruta especificada.")
    exit()

# Verificar que las columnas EJE_X y EJE_Y existen
if 'EJE_X' not in datos.columns or 'EJE_Y' not in datos.columns:
    print("Error: El archivo CSV debe contener las columnas 'EJE_X' y 'EJE_Y'.")
    exit()

# Seleccionar columnas de elementos (excluyendo EJE_X y EJE_Y)
columnas_elementos = [col for col in datos.columns if col not in ['EJE_X', 'EJE_Y']]

# Verificar que haya columnas de elementos
if not columnas_elementos:
    print("Error: No se encontraron columnas de elementos en el archivo CSV.")
    exit()

# Convertir columnas de elementos a tipo numérico, manejando valores no válidos
for col in columnas_elementos:
    datos[col] = pd.to_numeric(datos[col], errors='coerce')

# Verificar si hay columnas con datos no numéricos que resultaron en NaN
if datos[columnas_elementos].isna().any().any():
    print("Advertencia: Algunas columnas contienen valores no numéricos que se convirtieron a NaN.")
    print("Se imputarán los valores NaN con la mediana de cada columna.")
    for col in columnas_elementos:
        datos[col].fillna(datos[col].median(), inplace=True)

# Verificar que todas las columnas de elementos sean numéricas
if not all(datos[columnas_elementos].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    print("Error: Algunas columnas de elementos no son numéricas después de la conversión.")
    exit()

# Crear una columna de clasificación basada en los percentiles de los datos
# Usar percentiles igualmente espaciados para dar igual probabilidad a todas las clases
def clasificar_anomalia(fila, columnas):
    valores = fila[columnas]
    percentil_80 = valores.quantile(0.80)  # Para 5 (Fuerte)
    percentil_60 = valores.quantile(0.60)  # Para 4 (Moderada)
    percentil_40 = valores.quantile(0.40)  # Para 3 (Débil)
    percentil_20 = valores.quantile(0.20)  # Para 2 (Valor de Fondo)

    # Clasificación multivariable basada en el promedio de los valores
    promedio_valor = valores.mean()

    if promedio_valor >= percentil_80:
        return '5'
    elif promedio_valor >= percentil_60:
        return '4'
    elif promedio_valor >= percentil_40:
        return '3'
    elif promedio_valor >= percentil_20:
        return '2'
    else:
        return '1'

# Crear una columna temporal para entrenamiento
datos['Clasificacion_Temporal'] = datos[columnas_elementos].apply(
    lambda x: clasificar_anomalia(x, columnas_elementos), axis=1)

# Preparar los datos para el modelo
X = datos[columnas_elementos]
y = datos['Clasificacion_Temporal']

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest sin pesos personalizados para igualar probabilidades
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
modelo.fit(X_train, y_train)

# Predecir en todo el conjunto de datos
X_scaled_full = scaler.transform(X)
predicciones = modelo.predict(X_scaled_full)

# Añadir la columna de predicciones al DataFrame original
datos['ANOMALIA'] = predicciones

# Eliminar la columna temporal usada para entrenamiento
datos = datos.drop(columns=['Clasificacion_Temporal'])

# Generar la ruta de salida
ruta_salida = os.path.join(os.path.dirname(ruta_csv), 'datos_con_anomalias.csv')

# Guardar el nuevo CSV con la columna de predicciones
datos.to_csv(ruta_salida, index=False)
print(f"Archivo generado exitosamente en: {ruta_salida}")

# Mostrar la distribución de las predicciones
#print("\nDistribución de las predicciones:")
#print(datos['ANOMALIA'].value_counts())