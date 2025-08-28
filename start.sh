#!/bin/bash

# ==========================================
# Start script para Render - API Flask ML
# ==========================================

# Salir si ocurre algún error
set -e

# Mensaje de inicio
echo "Iniciando servicio Flask ML en Render..."

# Lanzar Gunicorn con 2 workers y timeout extendido
echo "Ejecutando Gunicorn..."
gunicorn api_modelo:app \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 120 \
    --log-level info

# Mensaje de servicio iniciado
echo "Servicio Flask ML ejecutándose en el puerto $PORT"


