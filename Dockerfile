FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/

# Copy the trained model into the image. This will fail the build if the
# model file is not present in the build context which makes the issue
# visible at build time rather than at runtime.
COPY api/alfix_model.pkl ./api/alfix_model.pkl

# Expose port and set MODEL_PATH env var so the app knows where to load the model
ENV MODEL_PATH=/app/api/alfix_model.pkl

EXPOSE 8080

CMD ["python", "api/app.py"]
