#!/bin/bash
gunicorn api_modelo:app --bind 0.0.0.0:$PORT --workers 2
