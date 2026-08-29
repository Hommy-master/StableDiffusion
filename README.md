# Stable Diffusion API

基于 Docker 的 Stable Diffusion 文生图 REST API 服务。镜像：`gogoshine/sd:latest`

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

### `GET /`

服务信息与接口概览。

### `GET /health`

健康检查。

**响应示例**

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

### `GET /output/<path>`

下载已生成的图片文件（对应容器内 `SD_OUTPUT_DIR`）。

当 `return_format=url` 时，`POST /txt2img` 返回的 URL 即指向此路径。

示例：`http://127.0.0.1:7860/output/20260829102000-42-0.png`

### `POST /txt2img`

根据文本提示词生成图片。

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
| `return_format` | string | 否 | `url` | 返回格式，见下表 |

**`return_format`**

| 值 | 说明 |
|---|---|
| `url` | 默认。图片写入 `output/`，响应中返回可下载 URL |
| `base64` | 响应中返回 `data:image/png;base64,...` |
| `file` | 仅当 `n_samples=1` 时生效，直接返回 PNG 二进制 |

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

**成功响应（`return_format=url`）**

```json
{
  "images": [
    "http://127.0.0.1:7860/output/20260829102000-42-0.png"
  ],
  "file_paths": [
    "/app/output/20260829102000-42-0.png"
  ],
  "parameters": {
    "prompt": "a painting of a virus monster playing guitar",
    "ddim_steps": 50,
    "scale": 7.5,
    "W": 512,
    "H": 512,
    "n_samples": 1,
    "seed": 42,
    "sampler": "ddim"
  },
  "has_nsfw": [false],
  "elapsed_seconds": 3.21
}
```

**错误响应**

| HTTP | 说明 |
|---|---|
| `400` | 请求体非 JSON，或缺少 `prompt` |
| `500` | 生成过程异常，`{"error": "..."}` |
