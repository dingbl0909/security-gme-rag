# Security GME RAG 轻量设计文档

## 1. 项目定位

`security-gme-rag` 是一个面向安防业务的轻量化多模态 RAG 系统，目标是在本地工作站上可运行、可演示，并能在简历和面试中讲清楚完整 AI 应用链路。

项目围绕设备接入、布控告警、部署排障和现场截图问答，将文本、图片和图文对统一抽象为知识单元，提供文搜文、文搜图、图搜文和图文联合检索能力。默认实现使用本地 Markdown、SQLite、BM25 和确定性向量模拟 GME 向量空间；生产增强时可以替换为 Dots.OCR、GME、Milvus、Redis、LangGraph 和 RAGAS。

## 2. 核心目标

- 支持安防文档、截图说明和历史案例的统一入库。
- 支持文本、图片、图文对三类知识单元。
- 支持 Dense + Sparse 混合检索。
- 支持会话上下文记忆和历史结论复用。
- 支持 Corrective RAG / Adaptive RAG 双层质量评估。
- 支持轻量 RAGAS 闭环记录，沉淀 query、answer、evidence 和指标。
- 支持低置信度结果触发人工审核。
- 提供 CLI 和 FastAPI 两种本地测试入口。

## 3. 目录结构

```text
security-gme-rag/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── chunker.py
│   ├── config.py
│   ├── database.py
│   ├── embedding.py
│   ├── evaluation.py
│   ├── index.py
│   ├── ingestion.py
│   ├── memory.py
│   ├── models.py
│   ├── parser.py
│   ├── retrieval.py
│   └── workflow.py
├── data/
│   ├── db/
│   ├── images/
│   └── security_docs/
├── docs/
│   └── design.md
├── scripts/
│   ├── ingest_demo.py
│   └── query_demo.py
├── README.md
├── requirements.txt
└── 项目介绍.txt
```

## 4. 模块设计

### 4.1 文档解析与数据流入

入库链路统一为 `app/ingest/`：

```text
DocumentSource.discover()
  -> DocumentParser.parse()   # Dots.OCR / Markdown
  -> ParsedDocument(blocks)
  -> RawDocument
  -> chunker / index
```

轻量版：

- `LocalCorpusSource` 扫描 `data/security_docs/`。
- `GME_RAG_PARSER_PROVIDER=local` 时，Markdown 走版面分块解析，其它格式生成可运行占位结果。
- 图片侧车文本仍由 `chunker.py` 在分块阶段合并为图文对。

生产增强：

- `GME_RAG_PARSER_PROVIDER=dots_ocr` 调用私有化 Dots.OCR 服务。
- 保留标题层级、段落、表格、图片 Base64 和 OCR 文本，输出 `LayoutBlock` 列表。

### 4.2 分块策略

`chunker.py` 使用两级分块：

- 章节标题粗分块。
- 超长内容按语义窗口二次切分。

图片知识单元会和相邻文本合并为图文对，避免图片和说明语义断裂。

### 4.3 多模态嵌入

`embedding.py` 抽象统一向量接口。

轻量版：

- 使用确定性 hash 向量模拟 GME embedding。
- 文本、图片描述和图文对进入同一向量空间。

生产增强：

- 替换为 `iic/gme-Qwen2-VL-7B-Instruct`。
- 使用 SentenceTransformers / vLLM 私有化部署。
- 通过 `GME_RAG_EMBEDDING_PROVIDER=gme`、`GME_API_KEY`、`GME_BASE_URL` 切换到远程 GME embedding API。

### 4.4 混合检索

`retrieval.py` 同时计算：

- Sparse：BM25 风格关键词得分。
- Dense：向量余弦相似度。
- Hybrid：按权重融合得分。

轻量版索引保存在 SQLite；生产版可替换为 Milvus Dense / Sparse 双索引。

当前代码提供 `GME_RAG_VECTOR_STORE=sqlite|milvus` 两种 provider。启用 Milvus 后，SQLite 仍保留 chunk 元数据和 sparse BM25 回退，Milvus 负责 dense 向量召回。

### 4.5 上下文记忆

`memory.py` 保存：

- 最近会话摘要。
- 关键结论。
- 人工反馈。

轻量版使用 SQLite；生产增强可以替换 Redis 短期记忆和 Milvus 长期语义记忆。

当前代码提供 `GME_RAG_MEMORY_PROVIDER=sqlite|redis` 两种 provider。启用 Redis 后，最近会话摘要和关键事实写入 Redis，同时仍落 SQLite 作为本地持久化备份。长期语义记忆可复用 Milvus collection 存储历史结论向量。

### 4.6 问答工作流

`workflow.py` 使用 LangGraph 实现可分支、可中断、可恢复的问答状态机：

```text
输入识别
  -> 历史上下文检索
  -> 图像检索 / 文本混合检索
  -> 证据质量评估
  -> Query 改写 / 重新检索
  -> 答案生成
  -> Response Relevancy / Faithfulness 评估
  -> 人工审批
  -> 从中断点恢复
```

纯图片问题优先进入图像检索链路；文本问题优先读取历史上下文并做混合检索。召回证据不足时先触发 Query 改写和重新检索；多次纠偏后仍不足，或答案不可靠时，进入 LangGraph `interrupt` 人工审批节点，并通过 `/resume` 从中断点恢复执行。

### 4.7 RAGAS 闭环

`ragas_loop.py` 实现轻量 RAGAS 风格评估闭环：

- `context_precision`
- `context_recall`
- `response_relevancy`
- `faithfulness`

每次答案生成后，系统会把 query、answer、evidence、metrics、是否需要人审和审核原因写入 `ragas_runs` 表。API 提供 `/ragas/runs` 查看近期评测记录，CLI 提供 `scripts/ragas_report.py` 查看本地评测报告。生产环境可将该模块替换为官方 RAGAS `evaluate()` 调用。

## 5. 本地可运行范围

本地版本做到：

- 初始化 SQLite 索引。
- 加载安防样例文档。
- 构建文本、图片、图文对知识单元。
- 执行混合检索。
- 输出证据、答案、评分和人工审核标记。
- 提供 `/health`、`/ingest`、`/query` API。

本地版本暂不内置：

- 真实 Dots.OCR 服务。
- 真实 GME 模型推理。
- 真实 Milvus / Redis 集群。
- 真实图片视觉理解。

这些能力保留接口，方便面试时讲清楚生产演进路径。

## 6. 简历表述建议

可以写成：

> 设计并实现面向安防场景的轻量化多模态 RAG 系统，统一组织设备接入手册、告警处置规范、部署文档和现场截图，抽象文本、图片、图文对三类知识单元；实现章节分块、BM25 + 向量混合检索、上下文记忆、Corrective / Adaptive RAG 质量评估和 Human-in-the-loop 审核流程，并预留 GME、Milvus、Redis、Dots.OCR 等生产级组件替换接口。

