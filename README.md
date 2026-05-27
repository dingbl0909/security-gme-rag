# Security GME RAG

面向安防场景的私有化多模态 RAG 系统：非结构化文档解析 → 章节/图文分块 → 混合检索 → LangGraph 问答与人工审核。

## 能力概览

- **数据流入**：`DocumentSource` → Dots.OCR 网关解析 → `ParsedDocument` → 分块索引
- **知识单元**：`text` / `image` / `text_image`
- **检索**：BM25 + 向量混合检索，文搜图 / 图搜文 / 图文联合
- **可靠性**：Corrective / Adaptive RAG 评分、RAGAS 记录、Human-in-the-loop
- **接口**：`/documents/upload`、`/ingest`、`/query`、`/resume`

## 仓库结构

```text
security-gme-rag/
├── app/                    # RAG 主服务
│   └── ingest/             # 采集、解析、入库管道
├── services/
│   └── dots_ocr_gateway/   # 本地解析网关（PDF/图片/HTML，可接 vLLM）
├── data/                   # 样例文档、图片侧车、上传目录
├── scripts/                # 启动、入库、冒烟测试
├── docker-compose.yml
└── docs/deploy.md          # 服务器部署说明
```

## 快速开始（本地）

```bash
git clone <your-repo-url> security-gme-rag
cd security-gme-rag
cp .env.example .env

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/start_local.sh

.venv/bin/python scripts/ingest_demo.py
.venv/bin/python scripts/query_demo.py "仓库北门摄像头离线，如何排查？"
```

- RAG API: http://127.0.0.1:8020/docs  
- 解析网关: http://127.0.0.1:8030/health  

停止服务：`bash scripts/start_local.sh stop`

## Docker 部署

```bash
cp .env.example .env
docker compose up -d --build
curl -X POST http://127.0.0.1:8020/ingest
```

详见 [docs/deploy.md](docs/deploy.md)。

## 主要 API

| 接口 | 说明 |
|------|------|
| `GET /health` | 服务与 parser 配置状态 |
| `POST /documents/upload?index=true` | 上传 PDF/图片等，解析并可入库 |
| `POST /ingest` | 重建样例知识库索引 |
| `POST /query` | 多模态问答 |
| `POST /resume` | 人工审核后继续 |

上传示例：

```bash
curl -X POST "http://127.0.0.1:8020/documents/upload?index=true" \
  -F "file=@manual.pdf"
```

查询示例：

```bash
curl -X POST http://127.0.0.1:8020/query \
  -H "Content-Type: application/json" \
  -d '{"query": "布控告警误报如何排查？", "thread_id": "demo-1"}'
```

## 配置说明

| 变量 | 默认 | 含义 |
|------|------|------|
| `GME_RAG_PARSER_PROVIDER` | `dots_ocr` | 使用解析网关 |
| `DOTS_OCR_BASE_URL` | `http://127.0.0.1:8030` | 网关地址 |
| `GME_RAG_EMBEDDING_PROVIDER` | `local` | 本地确定性向量；可改 `gme` |
| `GME_RAG_VECTOR_STORE` | `sqlite` | 可改 `milvus` |

解析网关：

| 变量 | 默认 | 含义 |
|------|------|------|
| `DOTS_GATEWAY_BACKEND` | `heuristic` | 无 GPU：PyMuPDF/Pillow 解析 |
| `DOTS_GATEWAY_BACKEND` | `vllm` | 有 GPU：转发至 vLLM dots.mocr |

## 生产增强

- **GME**：`GME_RAG_EMBEDDING_PROVIDER=gme`
- **Milvus / Redis**：见 `.env.example`
- **真实 Dots.OCR 模型**：vLLM 部署 + 网关 `DOTS_GATEWAY_BACKEND=vllm`

## 文档

- [部署指南](docs/deploy.md)
- [Dots.OCR 接入](docs/dots_ocr_integration.md)
- [设计说明](docs/design.md)
