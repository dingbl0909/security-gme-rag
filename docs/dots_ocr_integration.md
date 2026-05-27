# Dots.OCR 接入说明

## Dots.OCR 是什么

[Dots.OCR](https://github.com/rednote-hilab/dots.ocr)（现也有 **dots.mocr** 迭代）是小红书 HiLab 开源的**文档版面解析视觉语言模型**，不是 OpenAI 那种按量计费的公有云 API。

典型能力：

- PDF / 图片的多语言版面分析（标题、段落、表格、图片区域）
- 截图 OCR、网页版面解析
- 图表转 SVG 等（dots.mocr 系列）

官方推荐用 **vLLM 私有化部署**（需要 GPU），例如：

```bash
vllm serve rednote-hilab/dots.mocr \
  --trust-remote-code \
  --served-model-name dots-mocr
```

部署后得到的是 **OpenAI 兼容的 `/v1/chat/completions`**，或项目自带的 `dots_mocr/parser.py` 命令行工具。

## 你需要 API Key 还是本地部署？

| 方式 | 说明 |
|------|------|
| **本地 / 内网部署（推荐，与简历一致）** | 在 GPU 机器上用 vLLM 起模型服务；`DOTS_OCR_BASE_URL` 指向你封装的解析网关或 Sidecar。 |
| **API Key** | **不是 Dots.OCR 官方强制要求**。只有在你自己的网关前加了鉴权（Nginx、FastAPI 中间件）时，才在 `.env` 里配置 `DOTS_OCR_API_KEY`。 |
| **不部署 Dots.OCR** | 保持 `GME_RAG_PARSER_PROVIDER=local`：`.md` 仍可解析；上传的 PDF 等会走占位解析，便于先联调上传 API。 |

本仓库的 `DotsOcrParser` 默认调用你方提供的 **HTTP 解析网关**：

```http
POST {DOTS_OCR_BASE_URL}{DOTS_OCR_PARSE_PATH}
Content-Type: application/json
Authorization: Bearer <可选>

{
  "doc_id": "...",
  "filename": "manual.pdf",
  "mime_type": "application/pdf",
  "content_base64": "..."
}
```

期望响应（示例）：

```json
{
  "title": "设备接入手册",
  "blocks": [
    {"kind": "heading", "level": 2, "text": "适用场景"},
    {"kind": "paragraph", "text": "..."},
    {"kind": "image", "image_ref": "fig-1.png", "ocr_text": "...", "image_base64": "..."}
  ]
}
```

你需要写一个 **Thin Wrapper**（几十行 FastAPI），内部调用 vLLM 或 `dots_mocr/parser.py`，把结果转成上述 JSON。简历里的「vLLM 私有化部署解析服务」指的就是这一层。

## 与本项目 API 的关系

```text
客户端 --POST /documents/upload--> security-gme-rag
                                      |
                                      v
                               保存 data/uploads/
                                      |
                                      v
                               DotsOcrParser.parse()
                                      |
                    +-----------------+------------------+
                    | local / markdown fallback          | dots_ocr + DOTS_OCR_BASE_URL
                    v                                    v
              LayoutBlock 列表                    你的 Dots.OCR 网关
                    |
                    v
         index=true 时 chunk + 写入 SQLite/Milvus
```

### 调用示例

```bash
# 仅解析，不入库
curl -X POST "http://127.0.0.1:8020/documents/upload?index=false" \
  -F "file=@/path/to/alarm_manual.pdf"

# 解析并增量写入索引
curl -X POST "http://127.0.0.1:8020/documents/upload?index=true" \
  -F "file=@/path/to/alarm_manual.pdf"
```

`.env` 接入真实解析：

```bash
GME_RAG_PARSER_PROVIDER=dots_ocr
DOTS_OCR_BASE_URL=http://your-parser-service:8030
DOTS_OCR_PARSE_PATH=/parse
# 若网关需要鉴权再填写
DOTS_OCR_API_KEY=your-internal-token
```
