# ===========================================
# IDP System v2 - Backend Dockerfile
# Optimized for PaddleOCR & OpenCV
# ===========================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# libgomp1: Fixes the 'libgomp.so.1' missing error
# libgl1 & libglib2.0-0: Required for OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY main.py .
COPY models/ ./models/
COPY services/ ./services/
COPY utils/ ./utils/

# Create upload directory
RUN mkdir -p /app/uploads

# Set Environment Variables to prevent Paddle crashes
# Forces CPU if GPU is unavailable and prevents some optimization errors
ENV FLAGS_use_tensorrt=0
ENV DISABLE_MODEL_SOURCE_CHECK=True

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]