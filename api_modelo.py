from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# --- Ruta raíz de prueba ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "✅ API de predicción funcionando",
        "usar": "Haz POST a /predecir con los datos necesarios"
    })

# --- Cargar modelos entrenados ---
modelo_litros = joblib.load("modelo_litros_rf.joblib")
modelo_campo_seco = joblib.load("modelo_campo_seco_rf.joblib")
modelo_costo_agua = joblib.load("modelo_costo_agua_rf.joblib")
modelo_agua_desp = joblib.load("modelo_agua_desp_rf.joblib")
modelo_tiempo_riego = joblib.load("modelo_tiempo_riego_rf.joblib")

# --- Mapeo de cultivos ---
cultivos = {
    "CAÑA DE AZUCAR": 0,
    "MAIZ": 1,
    "ARROZ": 2,
    "ALFALFA": 3
}

# Variantes sin tilde
cultivos_normalizados = {
    "CANA DE AZUCAR": 0,
    "MAIZ": 1,
    "ARROZ": 2,
    "ALFALFA": 3
}

# --- Endpoint de predicción ---
@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()

    if not datos:
        return jsonify({"error": "No se recibió JSON válido"}), 400

    # Validar campos requeridos
    campos_requeridos = ["tipo_cultivo", "humedad_suelo", "temp_ambiente", "hum_ambiente"]
    for campo in campos_requeridos:
        if campo not in datos:
            return jsonify({"error": f"Falta campo: {campo}"}), 400

    try:
        # --- Cultivo ---
        tipo_cultivo = str(datos["tipo_cultivo"]).strip().upper()
        cod_cultivo = cultivos.get(tipo_cultivo) or cultivos_normalizados.get(tipo_cultivo)
        if cod_cultivo is None:
            return jsonify({"error": f"Cultivo no válido: {tipo_cultivo}"}), 400

        # --- Variables numéricas ---
        humedad_suelo = float(datos["humedad_suelo"])
        temp_ambiente = float(datos["temp_ambiente"])
        hum_ambiente = float(datos["hum_ambiente"])

        # --- Entrada para modelos ---
        entrada = [[humedad_suelo, cod_cultivo, temp_ambiente, hum_ambiente]]

        # --- Predicciones ---
        litros_estimados = modelo_litros.predict(entrada)[0]
        campo_seco = modelo_campo_seco.predict(entrada)[0]
        costo_agua = modelo_costo_agua.predict(entrada)[0]
        agua_desp = modelo_agua_desp.predict(entrada)[0]
        tiempo_riego = modelo_tiempo_riego.predict(entrada)[0]

        # --- Respuesta ---
        return jsonify({
            "litros_estimados": round(float(litros_estimados), 2),
            "campo_seco": "Sí" if campo_seco == 1 else "No",
            "costo_agua": round(float(costo_agua), 2),
            "agua_desp": round(float(agua_desp), 2),
            "tiempo_riego": round(float(tiempo_riego), 2)
        })

    except Exception as e:
        return jsonify({"error": f"Error al procesar datos: {str(e)}"}), 500


# --- Solo para desarrollo local ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)


