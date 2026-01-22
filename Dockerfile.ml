# ===========================================
# IDP System v2 - ML Service (GPU)
# ===========================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Detectron2 (GPU-safe)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Minimal API deps
RUN pip install fastapi uvicorn pillow opencv-python

COPY ml_service.py .

EXPOSE 9000
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "9000"]
