# Dockerfile — Stable Diffusion API
# Image: gogoshine/sd
#
# Build:
#   docker build -t gogoshine/sd .
#
# Run (GPU):
#   docker run --gpus all -p 7860:7860 gogoshine/sd
#
# Run (CPU, slow):
#   docker run -p 7860:7860 -e SD_DEVICE=cpu gogoshine/sd
#
# Run with a local model checkpoint (avoid re-downloading 4GB):
#   docker run --gpus all -p 7860:7860 \
#     -v /path/to/model.ckpt:/app/models/ldm/stable-diffusion-v1/model.ckpt \
#     gogoshine/sd
#
# Test:
#   curl -X POST http://localhost:7860/txt2img \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "a painting of a virus monster playing guitar", "n_samples": 1}'

FROM python:3.8-slim

LABEL org.opencontainers.image.title="Stable Diffusion API"
LABEL org.opencontainers.image.source="https://github.com/CompVis/stable-diffusion"
LABEL org.opencontainers.image.description="Stable Diffusion v1 text-to-image API server"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

# Install PyTorch with CUDA 11.3 support
# (The CUDA runtime is bundled in the wheel; the host only needs the NVIDIA driver)
RUN pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install git-based dependencies
RUN pip install --no-cache-dir git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    && pip install --no-cache-dir git+https://github.com/openai/CLIP.git@main#egg=clip

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . .
RUN pip install --no-cache-dir .

# Create model directory (checkpoint will be downloaded at runtime if absent)
RUN mkdir -p models/ldm/stable-diffusion-v1 outputs

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV SD_HOST=0.0.0.0
ENV SD_PORT=7860
ENV SD_DEVICE=cuda
ENV SD_SAMPLER=ddim
# Base URL for converting container paths (/app/...) into download URLs
# e.g. /app/outputs/test.png -> http://127.0.0.1/outputs/test.png
ENV DOWNLOAD_URL=http://127.0.0.1/

EXPOSE 7860

# Health check — verifies the Flask server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget -q -O- http://localhost:7860/health || exit 1

CMD ["python", "api.py"]
