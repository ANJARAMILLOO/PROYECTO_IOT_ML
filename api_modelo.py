from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Crear la app
app = Flask(__name__)

# === Cargar el modelo ===
modelo = joblib.load("modelo_entrenado.pkl")

# === Ruta de prueba (GET) ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"mensaje": "API de Predicción funcionando correctamente 🚀"})

# === Ruta para predecir (POST) ===
@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        # Obtener datos en JSON
        data = request.get_json(force=True)

        # Validar datos
        if not data:
            return jsonify({"error": "No se enviaron datos"}), 400

        # Convertir los valores en un DataFrame (asegúrate que las claves coincidan con las columnas que usaste para entrenar)
        df = pd.DataFrame([data])

        # Hacer la predicción
        prediccion = modelo.predict(df)

        # Respuesta
        return jsonify({
            "entrada": data,
            "prediccion": float(prediccion[0])  # Convertimos a float para que sea serializable en JSON
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Punto de entrada ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
