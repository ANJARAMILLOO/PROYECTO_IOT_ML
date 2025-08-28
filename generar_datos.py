import pandas as pd
import random

# === Información base de cultivos ===
# Valores aproximados obtenidos de literatura agrícola e IoT
cultivos_info = {
    "CAÑA DE AZUCAR": {"cod": 0, "litros_base": 110, "litros_ideales": 100, "umbral_seco": 60},
    "MAIZ":          {"cod": 1, "litros_base": 90,  "litros_ideales": 80,  "umbral_seco": 55},
    "ARROZ":         {"cod": 2, "litros_base": 140, "litros_ideales": 130, "umbral_seco": 70},
    "ALFALFA":       {"cod": 3, "litros_base": 85,  "litros_ideales": 75,  "umbral_seco": 65}
}

# === Columnas dataset ===
columnas = [
    'humedad_suelo', 'cultivo', 'temp_ambiente', 'hum_ambiente',
    'litros_requeridos', 'campo_seco', 'cultivo_cod',
    'precio_litro', 'litros_ideales', 'caudal_litros_min'
]

csv_path = "datos_entrenamiento_iot.csv"

# Crear CSV vacío con encabezados
pd.DataFrame(columns=columnas).to_csv(csv_path, index=False)

# === Generar datos ===
registros = []
for _ in range(150_000):
    cultivo = random.choice(list(cultivos_info.keys()))
    info = cultivos_info[cultivo]

    # --- Variables IoT (mismo rango que sensores ESP32) ---
    humedad_suelo = round(random.uniform(15, 90), 1)     # % humedad suelo
    temp_ambiente = round(random.uniform(12, 38), 1)     # °C
    hum_ambiente = round(random.uniform(35, 95), 1)      # % humedad relativa

    # --- Lógica litros requeridos ---
    # Menor humedad_suelo => más litros necesarios
    factor_deficit = max(0, (info["umbral_seco"] - humedad_suelo))
    litros_requeridos = info["litros_base"] + factor_deficit * random.uniform(1.0, 2.0)
    litros_requeridos = round(max(20, min(litros_requeridos, 280)), 1)

    # --- Campo seco ---
    campo_seco = 1 if humedad_suelo < info["umbral_seco"] else 0

    # --- Otros parámetros ---
    precio_litro = round(random.uniform(0.01, 0.05), 3)  # S/. por litro
    caudal = round(random.uniform(15, 35), 1)            # L/min
    litros_ideales = info["litros_ideales"]

    registros.append([
        humedad_suelo, cultivo, temp_ambiente, hum_ambiente,
        litros_requeridos, campo_seco, info["cod"],
        precio_litro, litros_ideales, caudal
    ])

# Guardar CSV final
pd.DataFrame(registros, columns=columnas).to_csv(csv_path, mode='a', header=False, index=False)

print(f"\n✅ Archivo generado exitosamente con {len(registros)} registros: {csv_path}")


