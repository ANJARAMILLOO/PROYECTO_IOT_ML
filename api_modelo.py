from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Crear la aplicación Flask
app = Flask(__name__)

# === Cargar el modelo entrenado ===
# Asegúrate de que modelo.pkl esté en el mismo directorio que app.py
try:
    modelo = joblib.load("modelo.pkl")
except Exception as e:
    modelo = None
    print(f"⚠️ No se pudo cargar el modelo: {e}")

# === Ruta raíz para verificar que la API funciona ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "✅ API de predicción funcionando",
        "usar": "Haz POST a /predecir con los datos necesarios"
    })

# === Ruta de predicción ===
@app.route("/predecir", methods=["POST"])
def predecir():
    if modelo is None:
        return jsonify({"error": "❌ Modelo no cargado en el servidor"}), 500

    try:
        # Recibir datos en formato JSON
        datos = request.get_json()

        # Validar que existan datos
        if not datos:
            return jsonify({"error": "No se recibieron datos"}), 400

        # Convertir a DataFrame para que coincida con el entrenamiento
        entrada = pd.DataFrame([datos])

        # Realizar predicción
        prediccion = modelo.predict(entrada)

        return jsonify({
            "entrada": datos,
            "prediccion": prediccion.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Configuración para Render / Gunicorn ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



