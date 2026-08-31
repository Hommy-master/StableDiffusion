"""
Stable Diffusion API Server
============================
A Flask-based HTTP API that wraps the txt2img generation pipeline.
Generation runs asynchronously on a background worker: submit a job,
receive a task_id, then poll for the result.

Usage:
    python api.py
    # POST  /txt2img          -> {"task_id": "<uuid>"}
    # GET   /tasks/<task_id>  -> status + result

Environment variables:
    SD_MODEL_PATH  - path to model checkpoint (default: models/ldm/stable-diffusion-v1/model.ckpt)
    SD_CONFIG      - path to model config yaml (default: configs/stable-diffusion/v1-inference.yaml)
    SD_HOST        - bind host (default: 0.0.0.0)
    SD_PORT        - bind port (default: 7860)
    SD_DEVICE      - device to use: cuda / cpu (default: auto-detect)
    SD_SAMPLER     - default sampler: ddim / plms / dpm_solver (default: ddim)
    SD_SKIP_SAFETY - set to "1" to disable NSFW safety checker (default: 0)
    HF_TOKEN       - HuggingFace token for downloading model checkpoint (optional)
    SD_MODEL_URL   - checkpoint download URL (default: hf-mirror.com mirror of SD 1.5 emaonly ckpt)
    SD_OUTPUT_DIR  - directory to save generated images (default: output)
    SD_MAX_TASKS   - max in-memory task records to retain (default: 500)
    DOWNLOAD_URL   - base URL used to convert container file paths into download URLs.
                     Rule: replace the container path prefix "/app/" with DOWNLOAD_URL.
                     e.g. DOWNLOAD_URL=http://127.0.0.1/
                          /app/output/test.png -> http://127.0.0.1/output/test.png
"""

import base64
import io
import os
import sys
import time
import traceback
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Lock, Thread

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup
# ---------------------------------------------------------------------------
MODEL = None
DEVICE = None
SAMPLERS = {}
SAFETY_CHECKER = None
SAFETY_FEATURE_EXTRACTOR = None

# ---------------------------------------------------------------------------
# Async task queue (single GPU worker)
# ---------------------------------------------------------------------------
TASK_QUEUE = Queue()
TASKS = {}
TASKS_LOCK = Lock()
WORKER_THREAD = None
MAX_TASKS = int(os.environ.get("SD_MAX_TASKS", "500"))

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
# Output directory & URL conversion
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("SD_OUTPUT_DIR", "output")
CONTAINER_ROOT = "/app/"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_output_dir():
    """Return the absolute output directory, creating it if necessary."""
    out_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def local_path_to_url(filepath):
    """
    Convert a container-local file path into a download URL.

    Replacement rule: container path prefix "/app/" -> DOWNLOAD_URL
        /app/output/test.png + DOWNLOAD_URL=http://127.0.0.1/
        -> http://127.0.0.1/output/test.png
    """
    download_url = os.environ.get("DOWNLOAD_URL", "http://127.0.0.1/")
    if not download_url.endswith("/"):
        download_url += "/"

    posix_path = filepath.replace(os.sep, "/")
    if not posix_path.startswith("/"):
        posix_path = os.path.abspath(filepath).replace(os.sep, "/")

    if posix_path.startswith(CONTAINER_ROOT):
        return download_url + posix_path[len(CONTAINER_ROOT):]

    return posix_path


def save_images(pil_images, seed, task_id=None):
    """Save PIL images to the output directory. Returns (file_paths, urls)."""
    out_dir = get_output_dir()
    timestamp = time.strftime("%Y%m%d%H%M%S")
    prefix = f"{task_id}-" if task_id else ""
    file_paths, urls = [], []
    for i, img in enumerate(pil_images):
        filename = f"{prefix}{timestamp}-{seed}-{i}.png"
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

        safety_model_id = os.environ.get(
            "SD_SAFETY_CHECKER_PATH", "CompVis/stable-diffusion-safety-checker"
        )
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
def _is_valid_ckpt(ckpt_path, min_bytes=3_000_000_000):
    """Sanity check: v1-5-pruned-emaonly.ckpt is ~4.27GB."""
    if not os.path.exists(ckpt_path):
        return False
    size = os.path.getsize(ckpt_path)
    if size < min_bytes:
        print(
            f"WARNING: {ckpt_path} is only {size / 1e9:.2f} GB "
            f"(< {min_bytes / 1e9:.1f} GB expected) - truncated or invalid, will re-download."
        )
        return False
    return True


def ensure_model_checkpoint(ckpt_path):
    """Download Stable Diffusion 1.5 (ema-only) checkpoint if not present."""
    if _is_valid_ckpt(ckpt_path):
        print(f"Model checkpoint found at {ckpt_path}")
        return True

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    url = os.environ.get(
        "SD_MODEL_URL",
        "https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
    )
    print(f"Checkpoint not found at {ckpt_path}. Downloading from {url} ...")

    import urllib.request

    headers = {"User-Agent": "StableDiffusion-API/1.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    part_path = ckpt_path + ".part"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(part_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 1024
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  {downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB ({pct}%)",
                        end="",
                        flush=True,
                    )
            print("\nDownload complete.")
        if not _is_valid_ckpt(part_path):
            raise IOError(f"downloaded file is truncated ({os.path.getsize(part_path)} bytes)")
        os.replace(part_path, ckpt_path)
        return True
    except Exception as e:
        print(f"\nERROR: failed to download model checkpoint: {e}")
        if os.path.exists(part_path):
            os.remove(part_path)
        print("Please manually download v1-5-pruned-emaonly.ckpt and place it at:")
        print(f"  {ckpt_path}")
        print("Or set SD_MODEL_PATH to point to an existing checkpoint.")
        return False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_app():
    global MODEL, DEVICE, SAMPLERS, SAFETY_CHECKER, SAFETY_FEATURE_EXTRACTOR, WORKER_THREAD

    config_path = os.environ.get("SD_CONFIG", "configs/stable-diffusion/v1-inference.yaml")
    ckpt_path = os.environ.get("SD_MODEL_PATH", "models/ldm/stable-diffusion-v1/model.ckpt")
    device_str = os.environ.get("SD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(device_str)
    print(f"Using device: {DEVICE}")

    if not ensure_model_checkpoint(ckpt_path):
        print("FATAL: no model checkpoint available. Exiting.")
        sys.exit(1)

    config = OmegaConf.load(config_path)
    MODEL = load_model_from_config(config, ckpt_path, DEVICE)

    SAFETY_CHECKER, SAFETY_FEATURE_EXTRACTOR = try_load_safety_checker(DEVICE)

    SAMPLERS["ddim"] = DDIMSampler(MODEL)
    SAMPLERS["plms"] = PLMSSampler(MODEL)
    SAMPLERS["dpm_solver"] = DPMSolverSampler(MODEL)
    print("Model ready. All samplers initialised.")

    WORKER_THREAD = Thread(target=worker_loop, name="sd-worker", daemon=True)
    WORKER_THREAD.start()
    print("Async generation worker started.")


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

                x_samples = MODEL.decode_first_stage(samples_ddim)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                x_checked, has_nsfw = check_safety(x_samples, DEVICE)

                pil_images = []
                for img_array in x_checked:
                    pil_images.append(
                        Image.fromarray((img_array * 255).round().astype(np.uint8))
                    )

    return pil_images, has_nsfw


# ---------------------------------------------------------------------------
# Task store & worker
# ---------------------------------------------------------------------------
def _prune_tasks_locked():
    """Drop oldest finished tasks when over MAX_TASKS. Caller must hold TASKS_LOCK."""
    if len(TASKS) <= MAX_TASKS:
        return
    finished = [
        (tid, t)
        for tid, t in TASKS.items()
        if t["status"] in (STATUS_SUCCEEDED, STATUS_FAILED)
    ]
    finished.sort(key=lambda item: item[1].get("finished_at") or item[1].get("created_at") or "")
    overflow = len(TASKS) - MAX_TASKS
    for tid, _ in finished[:overflow]:
        del TASKS[tid]


def create_task(params):
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "status": STATUS_PENDING,
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "queue_position": None,
        "parameters": params,
        "result": None,
        "error": None,
    }
    with TASKS_LOCK:
        TASKS[task_id] = task
        _prune_tasks_locked()
        pending_ahead = sum(1 for t in TASKS.values() if t["status"] == STATUS_PENDING)
        # This task is already pending; position is count of pending including itself
        task["queue_position"] = pending_ahead
    TASK_QUEUE.put(task_id)
    return dict(task)


def get_task(task_id):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            return None
        snapshot = dict(task)
        if snapshot["status"] == STATUS_PENDING:
            pending_ids = [
                tid
                for tid, t in TASKS.items()
                if t["status"] == STATUS_PENDING
            ]
            # Approximate FIFO by created_at
            pending_ids.sort(key=lambda tid: TASKS[tid]["created_at"])
            try:
                snapshot["queue_position"] = pending_ids.index(task_id) + 1
            except ValueError:
                snapshot["queue_position"] = None
        elif snapshot["status"] == STATUS_RUNNING:
            snapshot["queue_position"] = 0
        else:
            snapshot["queue_position"] = None
        return snapshot


def update_task(task_id, **fields):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            return
        task.update(fields)


def encode_images(images, urls, return_format):
    if return_format == "base64":
        result_images = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            result_images.append(f"data:image/png;base64,{b64}")
        return result_images
    return urls


def process_task(task_id):
    task = get_task(task_id)
    if task is None:
        return

    params = task["parameters"]
    update_task(
        task_id,
        status=STATUS_RUNNING,
        started_at=utc_now_iso(),
        queue_position=0,
    )
    print(f"[task {task_id}] running: {params.get('prompt', '')[:80]!r}")

    try:
        start = time.time()
        images, has_nsfw = generate(
            prompt=params["prompt"],
            ddim_steps=params["ddim_steps"],
            scale=params["scale"],
            W=params["W"],
            H=params["H"],
            n_samples=params["n_samples"],
            ddim_eta=params["ddim_eta"],
            seed=params["seed"],
            sampler_name=params["sampler"],
            precision=params["precision"],
        )
        elapsed = time.time() - start
        file_paths, urls = save_images(images, params["seed"], task_id=task_id)
        result_images = encode_images(images, urls, params["return_format"])

        update_task(
            task_id,
            status=STATUS_SUCCEEDED,
            finished_at=utc_now_iso(),
            queue_position=None,
            result={
                "images": result_images,
                "file_paths": file_paths,
                "has_nsfw": has_nsfw,
                "elapsed_seconds": round(elapsed, 2),
            },
            error=None,
        )
        print(f"[task {task_id}] succeeded in {elapsed:.2f}s")
    except Exception as e:
        traceback.print_exc()
        update_task(
            task_id,
            status=STATUS_FAILED,
            finished_at=utc_now_iso(),
            queue_position=None,
            result=None,
            error=str(e),
        )
        print(f"[task {task_id}] failed: {e}")
    finally:
        if DEVICE is not None and DEVICE.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def worker_loop():
    print("Worker loop entered; waiting for tasks.")
    while True:
        try:
            task_id = TASK_QUEUE.get(timeout=1.0)
        except Empty:
            continue
        try:
            process_task(task_id)
        finally:
            TASK_QUEUE.task_done()


def parse_txt2img_params(data):
    """Validate and normalise txt2img request body. Returns (params, error_response)."""
    if not data:
        return None, (jsonify({"error": "Request body must be JSON"}), 400)

    prompt = data.get("prompt")
    if not prompt:
        return None, (jsonify({"error": "Missing required field 'prompt'"}), 400)

    ddim_steps = int(data.get("ddim_steps", 50))
    scale = float(data.get("scale", 7.5))
    W = int(data.get("W", 512))
    H = int(data.get("H", 512))
    n_samples = int(data.get("n_samples", 1))
    ddim_eta = float(data.get("ddim_eta", 0.0))
    seed = int(data.get("seed", 42))
    sampler_name = data.get("sampler", os.environ.get("SD_SAMPLER", "ddim"))
    precision = data.get("precision", "autocast")
    return_format = data.get("return_format", "url")

    if return_format not in ("url", "base64"):
        return None, (
            jsonify({"error": "return_format must be 'url' or 'base64'"}),
            400,
        )

    ddim_steps = max(1, min(ddim_steps, 200))
    n_samples = max(1, min(n_samples, 8))
    W = max(64, min(W, 1024))
    H = max(64, min(H, 1024))

    params = {
        "prompt": prompt,
        "ddim_steps": ddim_steps,
        "scale": scale,
        "W": W,
        "H": H,
        "n_samples": n_samples,
        "ddim_eta": ddim_eta,
        "seed": seed,
        "sampler": sampler_name,
        "precision": precision,
        "return_format": return_format,
    }
    return params, None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve files from the output directory so returned URLs are downloadable."""
    return send_from_directory(os.path.abspath(OUTPUT_DIR), filename)


@app.route("/health", methods=["GET"])
def health():
    with TASKS_LOCK:
        pending = sum(1 for t in TASKS.values() if t["status"] == STATUS_PENDING)
        running = sum(1 for t in TASKS.values() if t["status"] == STATUS_RUNNING)
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else None,
        "queue": {
            "pending": pending,
            "running": running,
            "depth": TASK_QUEUE.qsize(),
        },
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Stable Diffusion API",
        "mode": "async",
        "endpoints": {
            "GET /health": "Health check",
            "POST /txt2img": "Submit txt2img task; returns task_id",
            "GET /tasks/<task_id>": "Query task status and result",
            "GET /output/<path>": "Download a generated image",
        },
        "usage": {
            "submit": {
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
            },
            "poll": {
                "url": "/tasks/<task_id>",
                "method": "GET",
            },
            "return_formats": {
                "url": "default — images saved to output/, result.images are download URLs",
                "base64": "result.images are data:image/png;base64,... URIs",
            },
            "statuses": [STATUS_PENDING, STATUS_RUNNING, STATUS_SUCCEEDED, STATUS_FAILED],
        },
    })


@app.route("/txt2img", methods=["POST"])
def txt2img():
    """Enqueue a generation task and return task_id immediately."""
    data = request.get_json(silent=True)
    params, err = parse_txt2img_params(data)
    if err is not None:
        return err

    task = create_task(params)
    return jsonify({
        "task_id": task["task_id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "queue_position": task["queue_position"],
    }), 202


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Return task status; includes result when succeeded, error when failed."""
    task = get_task(task_id)
    if task is None:
        return jsonify({"error": "task not found", "task_id": task_id}), 404

    body = {
        "task_id": task["task_id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "finished_at": task["finished_at"],
        "queue_position": task["queue_position"],
        "parameters": task["parameters"],
    }
    if task["status"] == STATUS_SUCCEEDED:
        body["result"] = task["result"]
    if task["status"] == STATUS_FAILED:
        body["error"] = task["error"]
    return jsonify(body)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_app()
    host = os.environ.get("SD_HOST", "0.0.0.0")
    port = int(os.environ.get("SD_PORT", "7860"))
    print(f"Starting API server on {host}:{port}")
    # threaded=True so clients can poll /tasks while the worker runs on GPU
    app.run(host=host, port=port, threaded=True)
