# 服务器本地部署指南

## 架构

```text
用户/前端
   -> RAG API (:8020)  security-gme-rag
         -> Dots.OCR Gateway (:8030)  解析 PDF/图片/HTML
         -> SQLite + 本地向量（默认）
```

## 方式一：脚本部署（推荐先验证）

```bash
git clone <your-repo-url> security-gme-rag
cd security-gme-rag
cp .env.example .env

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd services/dots_ocr_gateway
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../..

bash scripts/start_local.sh
.venv/bin/python scripts/ingest_demo.py
.venv/bin/python scripts/smoke_test.py
```

## 方式二：Docker Compose

```bash
cp .env.example .env
docker compose up -d --build
curl http://127.0.0.1:8030/health
curl http://127.0.0.1:8020/health
curl -X POST http://127.0.0.1:8020/ingest
```

## 上传非结构化文档

```bash
curl -X POST "http://127.0.0.1:8020/documents/upload?index=true" \
  -F "file=@/path/to/manual.pdf"
```

## 接入真实 Dots.OCR（GPU + vLLM）

1. 在 GPU 机器启动 vLLM：

```bash
vllm serve rednote-hilab/dots.mocr --trust-remote-code --served-model-name dots-mocr
```

2. 修改网关环境变量：

```bash
DOTS_GATEWAY_BACKEND=vllm
DOTS_VLLM_BASE_URL=http://<gpu-host>:8000
```

3. 重启 `dots-ocr-gateway` 与 `rag-api`。

## 防火墙与安全

- 仅内网开放 8020/8030
- 需要鉴权时设置 `DOTS_GATEWAY_API_KEY` 与 `DOTS_OCR_API_KEY`（值保持一致）
- `data/` 目录挂载持久化索引与上传文件
