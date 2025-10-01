FROM python:3.12-slim

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar código de la app
COPY app /app/app

# Directorios de trabajo
RUN mkdir -p /app/uploads /app/debug

ENV PORT=5000 \
    LOG_LEVEL=INFO \
    UPLOAD_DIR=/app/uploads

EXPOSE 5000

# Ejecutar servidor Flask por módulo
CMD ["python", "-m", "app.server"]
