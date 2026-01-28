# ===========================================
# IDP System v2 - Backend Dockerfile
# Supports Chandra OCR & PaddleOCR
# ===========================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN sed -i 's/main/main contrib non-free non-free-firmware/g' /etc/apt/sources.list.d/debian.sources || \
    seed -i 's/main/main contrib non-free/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    nvidia-cuda-toolkit \
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

# 1. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Pre-install Torch (Required by flash-attn for its build process)
# Using cu121 to match your GPU environment
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Copy and Install requirements (Remove flash-attn from requirements.txt first!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Attempt Flash Attention installation separately
# --no-build-isolation is key here; it tells pip to use the Torch we just installed
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Flash-attn build failed, continuing without it"

# 5. Ensure Uvicorn and Chandra are present
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