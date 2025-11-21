# Imagen base de Python
FROM python:3.10-slim

# Instalar dependencias del SO necesarias para LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Directorio de trabajo
WORKDIR /app

# Copiar requisitos y modelo
COPY requirements.txt .
COPY alfix_model.pkl . 

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la app
COPY api_analysis.py .

# Exponer puerto Cloud Run
EXPOSE 8080

# Run
CMD ["python", "api_analysis.py"]
