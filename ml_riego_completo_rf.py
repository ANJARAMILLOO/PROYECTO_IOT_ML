import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# === Mapeo de cultivos al mismo orden que el ESP32 ===
cultivo_map = {
    "CAÑA DE AZUCAR": 0,
    "MAIZ": 1,
    "ARROZ": 2,
    "ALFALFA": 3
}

# === Cargar dataset ===
df = pd.read_csv("datos_entrenamiento_iot.csv")

# Convertir nombre cultivo a código igual que el ESP32
if "cultivo" in df.columns:
    df["cultivo_cod"] = df["cultivo"].map(cultivo_map)

# === Features que vienen del ESP32 ===
X = df[["humedad_suelo", "cultivo_cod", "temp_ambiente", "hum_ambiente"]]

# === Targets derivados ===
df["costo_agua"] = df["litros_requeridos"] * df["precio_litro"]
df["agua_desperdiciada"] = (df["litros_requeridos"] - df["litros_ideales"]).clip(lower=0)
df["tiempo_riego"] = df["litros_requeridos"] / df["caudal_litros_min"]

targets = {
    "litros": df["litros_requeridos"],
    "campo_seco": df["campo_seco"],
    "costo_agua": df["costo_agua"],
    "desperdicio": df["agua_desperdiciada"],
    "tiempo_riego": df["tiempo_riego"]
}

# === Train/Test split ===
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# === Entrenar modelos ===
modelos = {}

def entrenar_y_guardar(modelo, X, y, filename):
    modelo.fit(X, y)
    joblib.dump(modelo, filename)
    return modelo

# 1. Litros
modelos["litros"] = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, targets["litros"].loc[X_train.index],
    "modelo_litros_rf.joblib"
)

# 2. Campo seco (clasificación)
modelos["campo_seco"] = entrenar_y_guardar(
    RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, targets["campo_seco"].loc[X_train.index],
    "modelo_campo_seco_rf.joblib"
)

# 3. Costo agua
modelos["costo"] = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, targets["costo_agua"].loc[X_train.index],
    "modelo_costo_agua_rf.joblib"
)

# 4. Desperdicio
modelos["desperdicio"] = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, targets["desperdicio"].loc[X_train.index],
    "modelo_agua_desp_rf.joblib"
)

# 5. Tiempo de riego
modelos["tiempo"] = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, targets["tiempo_riego"].loc[X_train.index],
    "modelo_tiempo_riego_rf.joblib"
)

print("\n✅ Modelos entrenados con las variables del ESP32 y guardados correctamente.")

