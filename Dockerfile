# Imagen base
FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el modelo
COPY api/ ./api/

# Exponer el puerto (Cloud Run usa $PORT)
EXPOSE 8080

# Comando de ejecución.
# Asumo que `api_analysis.py` expone `app = Flask(__name__)`
# Ajusta si tu archivo principal se llama distinto.
CMD ["python", "api/api_analysis.py"]
