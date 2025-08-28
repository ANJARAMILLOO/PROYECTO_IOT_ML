from flask import Flask, request, jsonify
import joblib
import numpy as np

# Crear la app
app = Flask(__name__)

# === Cargar modelos entrenados ===
modelos = {
    "litros": joblib.load("modelo_litros_rf.joblib"),
    "campo_seco": joblib.load("modelo_campo_seco_rf.joblib"),
    "costo": joblib.load("modelo_costo_agua_rf.joblib"),
    "desperdicio": joblib.load("modelo_agua_desp_rf.joblib"),
    "tiempo": joblib.load("modelo_tiempo_riego_rf.joblib"),
}

# === Mapeo de cultivos (igual al script de entrenamiento) ===
cultivo_map = {
    "CAÑA DE AZUCAR": 0,
    "MAIZ": 1,
    "ARROZ": 2,
    "ALFALFA": 3
}

# === Ruta de prueba ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "✅ API de predicción funcionando",
        "usar": "Haz POST a /predecir con los datos necesarios"
    })


# === Ruta de predicción ===
@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.get_json(force=True)

        # Validar que existan los campos esperados
        if not all(k in data for k in ["tipo_cultivo", "humedad_suelo", "temp_ambiente", "hum_ambiente"]):
            return jsonify({"error": "Faltan datos en el request"}), 400

        # Convertir cultivo a código
        cultivo_cod = cultivo_map.get(data["tipo_cultivo"].upper(), 1)  # Default MAIZ

        # Crear input para el modelo
        X = np.array([[ 
            float(data["humedad_suelo"]),
            cultivo_cod,
            float(data["temp_ambiente"]),
            float(data["hum_ambiente"])
        ]])

        # Hacer predicciones
        litros = modelos["litros"].predict(X)[0]
        campo_seco = modelos["campo_seco"].predict(X)[0]
        costo = modelos["costo"].predict(X)[0]
        desperdicio = modelos["desperdicio"].predict(X)[0]
        tiempo = modelos["tiempo"].predict(X)[0]

        # Construir respuesta
        return jsonify({
            "litros_estimados": round(float(litros), 2),
            "campo_seco": int(campo_seco),
            "costo_agua": round(float(costo), 2),
            "agua_desp": round(float(desperdicio), 2),
            "tiempo_riego": round(float(tiempo), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
