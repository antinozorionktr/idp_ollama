# ===========================================
# IDP System v2 - Backend Dockerfile
# Supports Chandra OCR & PaddleOCR
# ===========================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# build-essential & cmake: Required to compile C++ based python wheels
# libgomp1: Fixes the 'libgomp.so.1' missing error
# libgl1 & libglib2.0-0: Required for OpenCV
# poppler-utils: PDF processing for Chandra
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    tesseract-ocr \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Torch (GPU fallback to CPU)
# We handle the fallback here, but it MUST succeed in one of the two forms
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
    pip install --no-cache-dir torch torchvision

# Install dependencies
# Note: Removed "|| true" to ensure we catch installation errors
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install critical service components
RUN pip install --no-cache-dir uvicorn chandra-ocr

# Copy application code
COPY config.py .
COPY main.py .
COPY models/ ./models/
COPY services/ ./services/
COPY utils/ ./utils/

# Create upload directory
RUN mkdir -p /app/uploads

# Set Environment Variables
ENV FLAGS_use_tensorrt=0
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV CHANDRA_CACHE_DIR=/app/.cache/chandra

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/chandra

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application using the module flag -m
# This is more robust at finding the uvicorn executable
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]