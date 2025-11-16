FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the production model from Jenkins
COPY production_model.pkl ./model/production_model.pkl

# Copy source code
COPY src/ ./src/

# Create API endpoint
COPY app.py .

# Create non-root user for security
RUN useradd -m -u 1000 modeluser && chown -R modeluser:modeluser /app
USER modeluser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "app.py"]