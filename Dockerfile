FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod pillow numpy torchvision==0.20.1 timm huggingface_hub ultralytics==8.4.215
RUN SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

WORKDIR /app
COPY handler.py /app/handler.py

RUN python -c "import torchvision, ultralytics; from sam2.sam2_image_predictor import SAM2ImagePredictor; from handler import handler; assert 'lora_train_early_yolo' in handler({'input': {'task': 'capabilities'}})['training_tasks']"

# Modell beim Bauen vorladen, damit der erste Aufruf nicht wartet
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True)" || true

CMD ["python", "-u", "handler.py"]
