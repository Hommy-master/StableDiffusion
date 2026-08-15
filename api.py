"""
Stable Diffusion API Server
============================
A Flask-based HTTP API that wraps the txt2img generation pipeline.
The model is loaded once at startup and reused for all requests.

Usage:
    python api.py
    # then POST to http://localhost:7860/txt2img

Environment variables:
    SD_MODEL_PATH  - path to model checkpoint (default: models/ldm/stable-diffusion-v1/model.ckpt)
    SD_CONFIG      - path to model config yaml (default: configs/stable-diffusion/v1-inference.yaml)
    SD_HOST        - bind host (default: 0.0.0.0)
    SD_PORT        - bind port (default: 7860)
    SD_DEVICE      - device to use: cuda / cpu (default: auto-detect)
    SD_SAMPLER     - default sampler: ddim / plms / dpm_solver (default: ddim)
    SD_SKIP_SAFETY - set to "1" to disable NSFW safety checker (default: 0)
    HF_TOKEN       - HuggingFace token for downloading model checkpoint (optional)
    SD_MODEL_URL   - checkpoint download URL (default: hf-mirror.com mirror of sd-v1-4.ckpt)
    SD_OUTPUT_DIR  - directory to save generated images (default: outputs)
    DOWNLOAD_URL   - base URL used to convert container file paths into download URLs.
                     Rule: replace the container path prefix "/app/" with DOWNLOAD_URL.
                     e.g. DOWNLOAD_URL=http://127.0.0.1/
                          /app/outputs/test.png -> http://127.0.0.1/outputs/test.png
"""

import base64
import io
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from flask import Flask, request, jsonify

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup
# ---------------------------------------------------------------------------
MODEL = None
DEVICE = None
SAMPLERS = {}          # name -> sampler instance
SAFETY_CHECKER = None
SAFETY_FEATURE_EXTRACTOR = None


# ---------------------------------------------------------------------------
# Output directory & URL conversion
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("SD_OUTPUT_DIR", "outputs")
# Container root prefix used for path -> URL conversion
CONTAINER_ROOT = "/app/"


def get_output_dir():
    """Return the absolute output directory, creating it if necessary."""
    out_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def local_path_to_url(filepath):
    """
    Convert a container-local file path into a download URL.

    Replacement rule: container path prefix "/app/" -> DOWNLOAD_URL
        /app/outputs/test.png + DOWNLOAD_URL=http://127.0.0.1/
        -> http://127.0.0.1/outputs/test.png
    """
    download_url = os.environ.get("DOWNLOAD_URL", "http://127.0.0.1/")
    if not download_url.endswith("/"):
        download_url += "/"

    # Keep posix-style absolute paths (container paths) as-is;
    # only expand relative paths via abspath (host/local runs).
    posix_path = filepath.replace(os.sep, "/")
    if not posix_path.startswith("/"):
        posix_path = os.path.abspath(filepath).replace(os.sep, "/")

    if posix_path.startswith(CONTAINER_ROOT):
        return download_url + posix_path[len(CONTAINER_ROOT):]

    # Path is outside /app/ — return the raw path as a fallback
    return posix_path


def save_images(pil_images, seed):
    """Save PIL images to the output directory. Returns (file_paths, urls)."""
    out_dir = get_output_dir()
    timestamp = time.strftime("%Y%m%d%H%M%S")
    file_paths, urls = [], []
    for i, img in enumerate(pil_images):
        filename = f"{timestamp}-{seed}-{i}.png"
        filepath = os.path.join(out_dir, filename)
        img.save(filepath)
        file_paths.append(filepath)
        urls.append(local_path_to_url(filepath))
        print(f"Saved image: {filepath} -> {urls[-1]}")
    return file_paths, urls


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.to(device)
    model.eval()
    return model


def try_load_safety_checker(device):
    """Attempt to load the NSFW safety checker; return (checker, extractor) or (None, None)."""
    if os.environ.get("SD_SKIP_SAFETY", "0") == "1":
        print("Safety checker disabled via SD_SKIP_SAFETY=1")
        return None, None
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import (
            StableDiffusionSafetyChecker,
        )
        from transformers import AutoFeatureExtractor

        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        print("Loading safety checker...")
        extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
        checker.to(device)
        checker.eval()
        print("Safety checker loaded.")
        return checker, extractor
    except Exception as e:
        print(f"WARNING: could not load safety checker: {e}")
        print("The API will still work, but NSFW filtering is disabled.")
        return None, None


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in images]


def check_safety(x_image, device):
    """Run the safety checker if available; otherwise return images unchanged."""
    if SAFETY_CHECKER is None:
        return x_image, [False] * len(x_image)

    pil_images = numpy_to_pil(x_image)
    safety_checker_input = SAFETY_FEATURE_EXTRACTOR(pil_images, return_tensors="pt")
    safety_checker_input = {k: v.to(device) for k, v in safety_checker_input.items()}
    x_checked, has_nsfw = SAFETY_CHECKER(
        images=x_image, clip_input=safety_checker_input["pixel_values"]
    )
    return x_checked, has_nsfw


# ---------------------------------------------------------------------------
# Download model checkpoint if missing
# ---------------------------------------------------------------------------
def ensure_model_checkpoint(ckpt_path):
    """Download sd-v1-4.ckpt from HuggingFace if the checkpoint is not present."""
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1_000_000:
        print(f"Model checkpoint found at {ckpt_path}")
        return True

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    # Use hf-mirror.com by default (huggingface.co is unreachable in some regions).
    # Override with SD_MODEL_URL if you have another mirror / local file server.
    url = os.environ.get(
        "SD_MODEL_URL",
        "https://hf-mirror.com/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt",
    )
    print(f"Checkpoint not found at {ckpt_path}. Downloading from {url} ...")

    import urllib.request

    # Support optional HF token for private/gated models
    headers = {"User-Agent": "StableDiffusion-API/1.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(ckpt_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 1024  # 1 MB
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  {downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB ({pct}%)", end="", flush=True)
            print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\nERROR: failed to download model checkpoint: {e}")
        print("Please manually download sd-v1-4.ckpt and place it at:")
        print(f"  {ckpt_path}")
        print("Or set SD_MODEL_PATH to point to an existing checkpoint.")
        return False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_app():
    global MODEL, DEVICE, SAMPLERS, SAFETY_CHECKER, SAFETY_FEATURE_EXTRACTOR

    config_path = os.environ.get("SD_CONFIG", "configs/stable-diffusion/v1-inference.yaml")
    ckpt_path = os.environ.get("SD_MODEL_PATH", "models/ldm/stable-diffusion-v1/model.ckpt")
    device_str = os.environ.get("SD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(device_str)
    print(f"Using device: {DEVICE}")

    # Download checkpoint if needed
    if not ensure_model_checkpoint(ckpt_path):
        print("FATAL: no model checkpoint available. Exiting.")
        sys.exit(1)

    # Load model
    config = OmegaConf.load(config_path)
    MODEL = load_model_from_config(config, ckpt_path, DEVICE)

    # Load safety checker
    SAFETY_CHECKER, SAFETY_FEATURE_EXTRACTOR = try_load_safety_checker(DEVICE)

    # Pre-initialise samplers
    SAMPLERS["ddim"] = DDIMSampler(MODEL)
    SAMPLERS["plms"] = PLMSSampler(MODEL)
    SAMPLERS["dpm_solver"] = DPMSolverSampler(MODEL)
    print("Model ready. All samplers initialised.")


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------
def generate(
    prompt,
    ddim_steps=50,
    scale=7.5,
    W=512,
    H=512,
    C=4,
    f=8,
    n_samples=1,
    ddim_eta=0.0,
    seed=42,
    sampler_name="ddim",
    precision="autocast",
):
    """Generate images from a text prompt. Returns a list of PIL.Image."""
    seed_everything(seed)

    sampler = SAMPLERS.get(sampler_name, SAMPLERS["ddim"])
    batch_size = n_samples

    precision_scope = autocast if precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda" if DEVICE.type == "cuda" else "cpu"):
            with MODEL.ema_scope():
                # Encode prompt
                uc = None
                if scale != 1.0:
                    uc = MODEL.get_learned_conditioning(batch_size * [""])
                c = MODEL.get_learned_conditioning(batch_size * [prompt])

                shape = [C, H // f, W // f]
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=None,
                )

                # Decode to pixel space
                x_samples = MODEL.decode_first_stage(samples_ddim)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                # Safety check
                x_checked, has_nsfw = check_safety(x_samples, DEVICE)

                # Convert to PIL images
                pil_images = []
                for i, img_array in enumerate(x_checked):
                    pil_img = Image.fromarray((img_array * 255).round().astype(np.uint8))
                    pil_images.append(pil_img)

    return pil_images, has_nsfw


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else None,
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Stable Diffusion API",
        "endpoints": {
            "GET /health": "Health check",
            "POST /txt2img": "Generate image from text prompt",
        },
        "usage": {
            "url": "/txt2img",
            "method": "POST",
            "content_type": "application/json",
            "body_example": {
                "prompt": "a painting of a virus monster playing guitar",
                "ddim_steps": 50,
                "scale": 7.5,
                "W": 512,
                "H": 512,
                "n_samples": 1,
                "seed": 42,
                "sampler": "ddim",
                "return_format": "url",
            },
            "return_formats": {
                "url": "default — images saved to outputs/, returned as download URLs",
                "base64": "images returned inline as data URIs",
                "file": "single image returned directly as PNG binary",
            },
        },
    })


@app.route("/txt2img", methods=["POST"])
def txt2img():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing required field 'prompt'"}), 400

    # Parse optional parameters with defaults
    ddim_steps = int(data.get("ddim_steps", 50))
    scale = float(data.get("scale", 7.5))
    W = int(data.get("W", 512))
    H = int(data.get("H", 512))
    n_samples = int(data.get("n_samples", 1))
    ddim_eta = float(data.get("ddim_eta", 0.0))
    seed = int(data.get("seed", 42))
    sampler_name = data.get("sampler", os.environ.get("SD_SAMPLER", "ddim"))
    precision = data.get("precision", "autocast")
    return_format = data.get("return_format", "url")  # url | base64 | file

    # Clamp values to safe ranges
    ddim_steps = max(1, min(ddim_steps, 200))
    n_samples = max(1, min(n_samples, 8))
    W = max(64, min(W, 1024))
    H = max(64, min(H, 1024))

    try:
        start = time.time()
        images, has_nsfw = generate(
            prompt=prompt,
            ddim_steps=ddim_steps,
            scale=scale,
            W=W,
            H=H,
            n_samples=n_samples,
            ddim_eta=ddim_eta,
            seed=seed,
            sampler_name=sampler_name,
            precision=precision,
        )
        elapsed = time.time() - start

        # Always persist images to the output directory first
        file_paths, urls = save_images(images, seed)

        if return_format == "file" and len(images) == 1:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            buf.seek(0)
            from flask import send_file
            return send_file(buf, mimetype="image/png", download_name="generated.png")

        # Default: return download URLs (container path /app/ -> DOWNLOAD_URL)
        if return_format == "url":
            result_images = urls
        else:
            # base64-encoded images
            result_images = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                result_images.append(f"data:image/png;base64,{b64}")

        return jsonify({
            "images": result_images,
            "file_paths": file_paths,
            "parameters": {
                "prompt": prompt,
                "ddim_steps": ddim_steps,
                "scale": scale,
                "W": W,
                "H": H,
                "n_samples": n_samples,
                "seed": seed,
                "sampler": sampler_name,
            },
            "has_nsfw": has_nsfw,
            "elapsed_seconds": round(elapsed, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_app()
    host = os.environ.get("SD_HOST", "0.0.0.0")
    port = int(os.environ.get("SD_PORT", "7860"))
    print(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, threaded=False)
