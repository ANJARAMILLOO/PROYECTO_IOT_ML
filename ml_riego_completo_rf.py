import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
import joblib
import json

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
X_train, X_test, y_train_litros, y_test_litros = train_test_split(X, targets["litros"], test_size=0.2, random_state=42)
_, _, y_train_campo, y_test_campo = train_test_split(X, targets["campo_seco"], test_size=0.2, random_state=42)
_, _, y_train_costo, y_test_costo = train_test_split(X, targets["costo_agua"], test_size=0.2, random_state=42)
_, _, y_train_desp, y_test_desp = train_test_split(X, targets["desperdicio"], test_size=0.2, random_state=42)
_, _, y_train_tiempo, y_test_tiempo = train_test_split(X, targets["tiempo_riego"], test_size=0.2, random_state=42)

# === Entrenar modelos ===
metricas = {}

def entrenar_y_guardar(modelo, X_train, y_train, X_test, y_test, filename, es_clasificacion=False):
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, filename)

    # Evaluar precisión
    y_pred = modelo.predict(X_test)
    if es_clasificacion:
        score = accuracy_score(y_test, y_pred)
    else:
        score = r2_score(y_test, y_pred)

    return modelo, score

# 1. Litros (regresión)
_, score = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, y_train_litros,
    X_test, y_test_litros,
    "modelo_litros_rf.joblib"
)
metricas["litros"] = round(score * 100, 2)

# 2. Campo seco (clasificación)
_, score = entrenar_y_guardar(
    RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, y_train_campo,
    X_test, y_test_campo,
    "modelo_campo_seco_rf.joblib",
    es_clasificacion=True
)
metricas["campo_seco"] = round(score * 100, 2)

# 3. Costo agua (regresión)
_, score = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, y_train_costo,
    X_test, y_test_costo,
    "modelo_costo_agua_rf.joblib"
)
metricas["costo_agua"] = round(score * 100, 2)

# 4. Agua desperdiciada (regresión)
_, score = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, y_train_desp,
    X_test, y_test_desp,
    "modelo_agua_desp_rf.joblib"
)
metricas["desperdicio"] = round(score * 100, 2)

# 5. Tiempo de riego (regresión)
_, score = entrenar_y_guardar(
    RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42),
    X_train, y_train_tiempo,
    X_test, y_test_tiempo,
    "modelo_tiempo_riego_rf.joblib"
)
metricas["tiempo_riego"] = round(score * 100, 2)

# Guardar métricas en un JSON
with open("metricas_modelos.json", "w") as f:
    json.dump(metricas, f, indent=4)

print("\n✅ Modelos entrenados con las variables del ESP32 y guardados correctamente.")
print("📊 Precisión de cada modelo:", metricas)


