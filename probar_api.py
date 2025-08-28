import requests

# URL de tu API local (la que mostró Flask en VSCode)
url = "http://192.168.100.12:5000/predecir"

# Datos de prueba
data = {
    "tipo_cultivo": "TRIGO",
    "humedad_suelo": 50,
    "temp_ambiente": 25,
    "hum_ambiente": 55
}

# Hacer la petición POST
respuesta = requests.post(url, json=data)

# Mostrar la respuesta
print(respuesta.json())
