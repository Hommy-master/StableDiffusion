# Dockerfile — Stable Diffusion API
# Image: gogoshine/sd
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t gogoshine/sd .
#
# Or via compose (from docker/):
#   docker compose build
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
# NOTE: taming-transformers' setup.py builds an EMPTY distribution when installed
# the normal way (only dist-info metadata, no `taming` package). Upstream CompVis
# installs it editable (`pip install -e src/taming-transformers`), so we do the
# same: clone to /opt (persistent) and install editable. See environment.yaml.
RUN git clone https://github.com/CompVis/taming-transformers.git /opt/taming-transformers \
    && pip install --no-cache-dir -e /opt/taming-transformers \
    && pip install --no-cache-dir git+https://github.com/openai/CLIP.git@main#egg=clip

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . .
RUN pip install --no-cache-dir .

# Create model directory (checkpoint will be downloaded at runtime if absent)
RUN mkdir -p models/ldm/stable-diffusion-v1 output

# ---------------------------------------------------------------------------
# CLIP tokenizer/config — bake into the image so no runtime download is needed
# ---------------------------------------------------------------------------
# huggingface.co is unreachable from China; hf-mirror.com is used instead.
# The older transformers (4.19) in this image needs these files on disk
# (SD_CLIP_PATH) because hf-mirror lacks proper ETag/etag headers.
RUN mkdir -p /app/clip-vit-large-patch14 \
    && cd /app/clip-vit-large-patch14 \
    && BASE="https://hf-mirror.com/openai/clip-vit-large-patch14/resolve/main" \
    && wget -q "$BASE/vocab.json" \
    && wget -q "$BASE/merges.txt" \
    && wget -q "$BASE/tokenizer_config.json" \
    && wget -q "$BASE/special_tokens_map.json" \
    && wget -q "$BASE/config.json" \
    && ls -la /app/clip-vit-large-patch14

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV SD_HOST=0.0.0.0
ENV SD_PORT=7860
ENV SD_DEVICE=cuda
ENV SD_SAMPLER=ddim
# CLIP tokenizer/config baked into the image at build time
ENV SD_CLIP_PATH=/app/clip-vit-large-patch14
# Skip NSFW safety checker by default: transformers 4.19.2 cannot download it
# from hf-mirror.com (etag/resolve-cache URL bug) and huggingface.co is
# unreachable from China. Set SD_SKIP_SAFETY=0 + SD_SAFETY_CHECKER_PATH=<dir>
# to enable it with a local copy of CompVis/stable-diffusion-safety-checker.
ENV SD_SKIP_SAFETY=1
# Use the HuggingFace mirror for any remaining runtime downloads
ENV HF_ENDPOINT=https://hf-mirror.com
# Base URL for converting container paths (/app/...) into download URLs
# e.g. /app/output/test.png -> http://127.0.0.1:7860/output/test.png
ENV DOWNLOAD_URL=http://127.0.0.1:7860/

EXPOSE 7860

# Health check — verifies the Flask server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget -q -O- http://localhost:7860/health || exit 1

CMD ["python", "api.py"]
