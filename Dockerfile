FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod pillow numpy torchvision timm

WORKDIR /app
COPY handler.py /app/handler.py

# Modell beim Bauen vorladen, damit der erste Aufruf nicht wartet
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True)" || true

CMD ["python", "-u", "handler.py"]
