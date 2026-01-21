FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY config.py .
COPY main.py .
COPY models/ ./models/
COPY services/ ./services/

# Ensure utils exists
RUN mkdir -p /app/utils
COPY utils/ ./utils/

# Uploads
RUN mkdir -p /app/uploads

# Expose NEW port
EXPOSE 8002

# Health
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
