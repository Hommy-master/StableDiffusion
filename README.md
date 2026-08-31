# Stable Diffusion API

基于 Docker 的 Stable Diffusion 文生图 REST API 服务。镜像：`gogoshine/sd:latest`

生成接口为**异步**模式：提交任务后立即返回 `task_id`，再通过 `task_id` 查询状态与结果，避免长时间阻塞 HTTP 连接。

## 容器部署

### 前置条件

- Docker / Docker Compose
- NVIDIA GPU 及 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)（CPU 模式可用但极慢）

### 使用 Compose 启动

```bash
cd docker
docker compose up -d
```

默认映射端口 `7860`，服务地址：`http://127.0.0.1:7860`

### 目录挂载

| 宿主机路径 | 容器路径 | 说明 |
|---|---|---|
| `docker/models` | `/app/models` | 模型权重（避免重复下载约 4GB） |
| `docker/output` | `/app/output` | 生成图片输出目录 |
| `docker/hf-cache` | `/root/.cache/huggingface` | HuggingFace 缓存 |

建议将 checkpoint 放到：

```text
docker/models/ldm/stable-diffusion-v1/model.ckpt
```

若该路径不存在，容器启动时会按 `SD_MODEL_URL` 自动下载。

### 常用环境变量

在 `docker/docker-compose.yaml` 中配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SD_HOST` / `SD_PORT` | `0.0.0.0` / `7860` | 监听地址与端口 |
| `SD_DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `SD_SAMPLER` | `ddim` | 默认采样器：`ddim` / `plms` / `dpm_solver` |
| `SD_MODEL_PATH` | `/app/models/ldm/stable-diffusion-v1/model.ckpt` | 权重路径 |
| `SD_MODEL_URL` | hf-mirror 上的 sd-v1-4.ckpt | 权重缺失时的下载地址 |
| `SD_SKIP_SAFETY` | `1` | `1` 跳过 NSFW 安全检查器 |
| `SD_OUTPUT_DIR` | `output` | 容器内输出目录 |
| `SD_MAX_TASKS` | `500` | 内存中保留的任务记录上限 |
| `DOWNLOAD_URL` | `http://127.0.0.1:7860/` | 将 `/app/` 路径替换为该前缀，生成可下载 URL |
| `HF_TOKEN` | （空） | 下载受限模型时可选 |

### 仅拉取镜像运行

```bash
docker pull gogoshine/sd:latest
docker run --gpus all -p 7860:7860 \
  -v "$(pwd)/docker/models:/app/models" \
  -v "$(pwd)/docker/output:/app/output" \
  -e DOWNLOAD_URL=http://127.0.0.1:7860/ \
  gogoshine/sd:latest
```

---

## REST API

Base URL：`http://127.0.0.1:7860`

调用流程：

1. `POST /txt2img` 提交任务 → 获得 `task_id`
2. 轮询 `GET /tasks/<task_id>`，直到 `status` 为 `succeeded` 或 `failed`
3. 成功时从 `result.images` 取下载 URL（或 base64），或用 `GET /output/<path>` 下载文件

任务状态：`pending` → `running` → `succeeded` | `failed`  
GPU 上由**单个后台 worker**串行执行，多任务会排队。

### `GET /`

服务信息与接口概览。

### `GET /health`

健康检查。

**响应示例**

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "queue": {
    "pending": 1,
    "running": 1,
    "depth": 1
  }
}
```

### `GET /output/<path>`

下载已生成的图片文件（对应容器内 `SD_OUTPUT_DIR`）。

示例：`http://127.0.0.1:7860/output/<task_id>-20260829102000-42-0.png`

### `POST /txt2img`

提交文生图任务，立即返回 `task_id`（不阻塞等待生成完成）。

**Headers**

```http
Content-Type: application/json
```

**请求体**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | string | 是 | — | 文本提示词 |
| `ddim_steps` | int | 否 | `50` | 采样步数，范围 1–200 |
| `scale` | float | 否 | `7.5` | classifier-free guidance scale |
| `W` | int | 否 | `512` | 宽度（像素），范围 64–1024 |
| `H` | int | 否 | `512` | 高度（像素），范围 64–1024 |
| `n_samples` | int | 否 | `1` | 生成数量，范围 1–8 |
| `ddim_eta` | float | 否 | `0.0` | DDIM eta（0 为确定性） |
| `seed` | int | 否 | `42` | 随机种子 |
| `sampler` | string | 否 | 环境变量 `SD_SAMPLER` | `ddim` / `plms` / `dpm_solver` |
| `precision` | string | 否 | `autocast` | `autocast` 或 `full` |
| `return_format` | string | 否 | `url` | `url` 或 `base64`（写入任务结果） |

**请求示例**

```bash
curl -X POST http://127.0.0.1:7860/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a painting of a virus monster playing guitar",
    "ddim_steps": 50,
    "scale": 7.5,
    "W": 512,
    "H": 512,
    "n_samples": 1,
    "seed": 42,
    "sampler": "ddim",
    "return_format": "url"
  }'
```

**响应 `202 Accepted`**

```json
{
  "task_id": "3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f",
  "status": "pending",
  "created_at": "2026-08-31T08:00:00+00:00",
  "queue_position": 1
}
```

**错误响应**

| HTTP | 说明 |
|---|---|
| `400` | 请求体非 JSON、缺少 `prompt`，或 `return_format` 非法 |

### `GET /tasks/<task_id>`

查询任务状态与结果。

**请求示例**

```bash
curl http://127.0.0.1:7860/tasks/3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f
```

**进行中**

```json
{
  "task_id": "3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f",
  "status": "running",
  "created_at": "2026-08-31T08:00:00+00:00",
  "started_at": "2026-08-31T08:00:01+00:00",
  "finished_at": null,
  "queue_position": 0,
  "parameters": {
    "prompt": "a painting of a virus monster playing guitar",
    "ddim_steps": 50,
    "scale": 7.5,
    "W": 512,
    "H": 512,
    "n_samples": 1,
    "ddim_eta": 0.0,
    "seed": 42,
    "sampler": "ddim",
    "precision": "autocast",
    "return_format": "url"
  }
}
```

**成功 `succeeded`**

```json
{
  "task_id": "3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f",
  "status": "succeeded",
  "created_at": "2026-08-31T08:00:00+00:00",
  "started_at": "2026-08-31T08:00:01+00:00",
  "finished_at": "2026-08-31T08:01:10+00:00",
  "queue_position": null,
  "parameters": { "...": "..." },
  "result": {
    "images": [
      "http://127.0.0.1:7860/output/3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f-20260829102000-42-0.png"
    ],
    "file_paths": [
      "/app/output/3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f-20260829102000-42-0.png"
    ],
    "has_nsfw": [false],
    "elapsed_seconds": 68.5
  }
}
```

**失败 `failed`**

```json
{
  "task_id": "3f1c8b2a-9d4e-4a1f-8c2b-1a2b3c4d5e6f",
  "status": "failed",
  "error": "CUDA out of memory",
  "parameters": { "...": "..." }
}
```

**错误响应**

| HTTP | 说明 |
|---|---|
| `404` | `task_id` 不存在（或进程重启后已丢失；任务仅存于内存） |

### 轮询示例

```bash
TASK_ID=$(curl -s -X POST http://127.0.0.1:7860/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat","n_samples":1}' | python -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

while true; do
  RESP=$(curl -s "http://127.0.0.1:7860/tasks/$TASK_ID")
  STATUS=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "status=$STATUS"
  case "$STATUS" in
    succeeded|failed) echo "$RESP"; break ;;
  esac
  sleep 2
done
```
