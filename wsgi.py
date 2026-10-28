"""
wsgi.py — Production WSGI Entrypoint for LangBill AI
Used by production application servers:
  - Waitress (Windows / Cross-platform):
      waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
  - Gunicorn (Linux / Container):
      gunicorn --workers 4 --threads 4 --bind 0.0.0.0:5000 wsgi:app
"""
import os
import sys

import config
from app import app

if __name__ == "__main__":
    try:
        from waitress import serve
        print(f"Starting production server (Waitress) on http://{config.HOST}:{config.PORT} ...")
        serve(app, host=config.HOST, port=config.PORT)
    except ImportError:
        print(f"Waitress not installed. Starting server with config settings on http://{config.HOST}:{config.PORT} ...")
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

