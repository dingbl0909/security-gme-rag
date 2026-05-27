# Security GME RAG 代码理解与面试讲解

## 1. 项目整体定位

`security-gme-rag` 可以理解成一个“本地可跑的生产级多模态 RAG 缩小版”。它的核心价值不是简单问答，而是把安防知识从文档、截图、图文案例里统一组织起来，再通过混合检索、上下文记忆和质量评估给出可追溯回答。

主链路如下：

```text
安防资料
  -> parser.py 解析为 RawDocument
  -> chunker.py 切成 text / image / text_image 知识单元
  -> embedding.py 生成统一向量
  -> index.py 写入 SQLite / Milvus
  -> retrieval.py 做 BM25 + Dense 混合检索
  -> workflow.py 生成答案、评估质量、判断是否人工审核
```

## 2. 核心数据结构

核心数据结构在 `app/models.py`。

### RawDocument

`RawDocument` 表示原始文档解析后的统一对象：

```python
@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    title: str
    source: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
```

它用于承接 Markdown、PDF、OCR 解析结果等不同来源的资料。

### KnowledgeChunk

`KnowledgeChunk` 是项目中最关键的知识单元抽象：

```python
@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    modality: Modality
    content: str
    image_ref: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
```

其中 `modality` 支持：

- `text`：纯文本知识。
- `image`：图片知识，当前用图片描述 mock。
- `text_image`：图文对，例如“截图 + 相邻 SOP 文本”。

这个抽象让文本、图片描述和图文对可以进入同一条 RAG 链路。

### RetrievalHit

`RetrievalHit` 表示一次检索命中的结果：

```python
@dataclass(frozen=True)
class RetrievalHit:
    chunk: KnowledgeChunk
    dense_score: float
    sparse_score: float
    hybrid_score: float
```

它同时保留 dense、sparse 和 hybrid 分数，方便解释召回结果。

## 3. 入库链路

入库入口在 `app/ingestion.py`：

```python
def ingest_demo_corpus() -> int:
    settings = get_settings()
    documents = load_documents(settings.docs_dir)
    chunks = chunk_documents(documents)
    return LocalIndex().rebuild(chunks)
```

这对应完整的入库流程：

```text
data/security_docs/*.md
  -> parser.py 解析 RawDocument
  -> chunker.py 切成 KnowledgeChunk
  -> embedding.py 生成向量
  -> index.py 写入 SQLite / Milvus
```

面试中可以这样讲：

> 我先把非结构化资料统一解析成文档对象，再做章节分块和图文对构建，最后进入统一向量空间和混合检索索引。

## 4. 分块策略

分块逻辑在 `app/chunker.py`。

当前轻量版做了两件事：

- 按 `##` 标题进行章节级粗分块。
- 对超长内容进行窗口切分。
- 如果发现 Markdown 图片引用，就把图片描述和相邻文本合并成 `text_image` chunk。

这样做的原因是：安防资料里经常有截图、表格、配置项和上下文说明。如果简单按固定长度切分，图片和说明文字可能被拆开，导致检索时丢失语义。

例如“大厅误报截图”会和“布控告警误报分析规范”的处置建议进入同一个图文块。

## 5. 多模态 Embedding 层

Embedding 层在 `app/embedding.py`。

### 本地默认实现

默认使用 `GmeLikeEmbedder`：

```python
class GmeLikeEmbedder:
    """Deterministic local substitute for a GME multimodal embedder."""
```

它不是真实 GME，而是确定性 hash 向量。这样做的目的是让项目在没有 GPU、没有 GME 服务的情况下也能完整跑通。

### 生产增强实现

生产增强使用 `GmeApiEmbedder`：

```python
class GmeApiEmbedder:
    """HTTP adapter for a private GME Qwen2-VL embedding service."""
```

它通过以下配置启用：

```bash
GME_RAG_EMBEDDING_PROVIDER=gme
GME_API_KEY=your_api_key
GME_BASE_URL=http://your-gme-service/v1
GME_MODEL=iic/gme-Qwen2-VL-7B-Instruct
```

Provider 选择逻辑：

```python
def build_embedder(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.embedding_provider == "local":
        return GmeLikeEmbedder()
    if settings.embedding_provider == "gme":
        return GmeApiEmbedder(settings)
```

面试重点：

> 我没有把系统绑死在 mock 或某个模型上，而是通过 provider 抽象实现“本地可跑、生产可替换”。

## 6. 混合检索

检索核心在 `app/retrieval.py`。

检索时会同时计算：

- `dense_score`：向量相似度，生产可走 Milvus。
- `sparse_score`：BM25 关键词匹配。
- `hybrid_score`：两者加权融合。

核心逻辑：

```python
query_vector = self.embedder.embed(query, image_ref)
milvus_dense_scores = self.vector_store.search(query_vector, top_k or self.settings.top_k)
query_tokens = tokenize(query)
```

每个 chunk 计算分数：

```python
dense_score = milvus_dense_scores.get(
    chunk.chunk_id,
    max(cosine_similarity(query_vector, vector), 0.0),
)
sparse_score = bm25_score(query_tokens, chunk, document_frequencies, len(indexed))
hybrid_score = self.settings.dense_weight * dense_score + self.settings.sparse_weight * sparse_score
```

如果用户传入 `image_ref`，说明这是图文联合检索，会对匹配图片的图文块加权：

```python
if image_ref and chunk.image_ref == image_ref:
    hybrid_score += 0.5
```

面试重点：

> Sparse 解决关键词精确匹配，Dense 解决语义召回，Hybrid 融合兼顾准确率和召回率。对于图文查询，我会对同图像引用的图文 chunk 做额外加权。

## 7. Milvus 适配

Milvus 相关代码在 `app/vector_store.py`。

系统提供两个 vector store provider：

```python
def build_vector_store(settings: Settings, database: RagDatabase):
    if settings.vector_store == "sqlite":
        return SQLiteVectorStore(database)
    if settings.vector_store == "milvus":
        return MilvusVectorStore(settings)
```

默认 `SQLiteVectorStore` 是 no-op，因为向量已经保存在 SQLite 中，方便本地演示。

生产模式下启用 Milvus：

```bash
GME_RAG_VECTOR_STORE=milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=security_gme_rag_chunks
```

设计思路：

- SQLite：保存 chunk 元数据、本地向量和 BM25 回退。
- Milvus：负责 dense 向量召回。

这样即使 Milvus 不可用，系统仍能保留本地回退能力。

## 8. 上下文记忆

记忆层在 `app/memory.py`。

默认使用 SQLite 保存：

- 最近会话摘要。
- 关键事实。

如果配置 Redis：

```bash
GME_RAG_MEMORY_PROVIDER=redis
REDIS_URL=redis://localhost:6379/0
```

系统会优先从 Redis 读取短期记忆：

```python
if self.redis is not None:
    payload = self.redis.get(self._key(thread_id))
    if payload:
        data = json.loads(payload)
        return data.get("summary", ""), data.get("facts", [])
```

保存时同时写 Redis 和 SQLite：

```python
if self.redis is not None:
    self.redis.set(..., ex=60 * 60 * 24)
self.database.save_memory(thread_id, summary, merged_facts)
```

面试可以这样讲：

> Redis 用于短期会话记忆，降低多轮问答重复检索成本；SQLite 作为本地持久化备份。后续长期语义记忆可以写入 Milvus 的历史结论 collection。

## 9. LangGraph 问答工作流

主工作流在 `app/workflow.py`，现在已经使用 LangGraph 实现可分支、可中断、可恢复的状态机。

核心节点：

```text
START
  -> classify_input
  -> load_memory
  -> retrieve_image / retrieve_text
  -> grade_context
  -> rewrite_query / generate_answer / human_review
  -> grade_answer
  -> human_review / save_memory
  -> END
```

路由逻辑：

- 纯图片或带 `image_ref` 的问题进入 `retrieve_image`。
- 文本问题进入 `retrieve_text`。
- 证据不足时先进入 `rewrite_query` 做一次 Query 改写和重检。
- 重检后仍不足，或答案不可靠，进入 `human_review`。
- 人审通过或驳回后，通过 `/resume` 从中断点继续执行并写回记忆。

这个设计可以在面试中说明：不是简单线性 RAG，而是具备分支路由、纠错重试和 Human-in-the-loop 恢复能力。

## 10. Corrective / Adaptive RAG

评估逻辑在 `app/evaluation.py`，工作流中调用：

```python
precision = context_precision(query, hits)
recall = context_recall(query, hits)
relevancy = response_relevancy(query, answer)
faithful = faithfulness(answer, hits)
```

然后通过 `review_decision()` 判断是否需要人工审核：

```python
if not hits:
    return True, "未召回证据。"
if precision < 0.35 or recall < 0.2:
    return True, "检索证据覆盖不足，需要 Query 改写或人工补充资料。"
if faithful < 0.25:
    return True, "回答与证据一致性偏低，需要人工审核。"
```

这就是轻量版的 Corrective / Adaptive RAG：

- 检索证据不足时，触发 Query 改写或人工补充资料。
- 回答与用户问题相关性低时，触发人工审核。
- 回答与证据一致性低时，触发人工审核。
- 不只是生成答案，而是判断答案是否可信。

## 11. RAGAS 闭环

`app/ragas_loop.py` 实现了轻量 RAGAS 风格闭环。它不会强依赖官方 `ragas` 包，而是先把生产中需要关注的评估样本和指标落地：

- query
- answer
- evidence
- context_precision
- context_recall
- response_relevancy
- faithfulness
- needs_review
- review_reason

每次工作流完成答案评估后，都会调用 `RagasEvaluationLoop.record()` 写入 `ragas_runs` 表。这样可以持续观察哪些问题召回弱、哪些回答一致性差、哪些场景频繁进入人工审核。

查看近期评估记录：

```bash
python scripts/ragas_report.py --limit 10
```

API 入口：

```text
GET /ragas/runs
```

面试可以这样讲：

> 当前项目实现了轻量 RAGAS 闭环，把检索、生成、人审相关指标持久化下来，用于后续分析和优化；生产环境可以把 `RagasEvaluationLoop.evaluate()` 替换为官方 RAGAS `evaluate()`。

## 12. API 和 CLI

CLI：

```bash
python scripts/ingest_demo.py
python scripts/query_demo.py "仓库北门摄像头离线，如何排查？"
python scripts/query_demo.py "根据大厅误报截图，分析布控告警为什么误报" --image-ref lobby_false_alarm.txt
```

API：

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8020
```

接口包括：

- `GET /health`
- `POST /ingest`
- `POST /query`

## 13. 面试讲解顺序

建议按下面顺序讲：

1. 业务问题：安防知识分散在文档、截图、告警案例里，传统关键词检索不够。
2. 知识建模：统一抽象成 `text / image / text_image` 三类 chunk。
3. 入库链路：解析 -> 分块 -> embedding -> SQLite / Milvus。
4. 检索链路：BM25 sparse + dense vector hybrid。
5. 记忆链路：Redis 短期记忆，SQLite 回退，Milvus 长期记忆预留。
6. 质量控制：context precision / recall / response relevancy / faithfulness，低置信度 human review。
7. 工程化：provider 化设计，本地可跑，生产可替换 GME / Milvus / Redis。

## 14. 简历表述

可以写成：

> 设计并实现面向安防场景的轻量化多模态 RAG 系统，统一组织设备接入手册、告警处置规范、部署文档和现场截图，抽象文本、图片、图文对三类知识单元；实现章节分块、GME 风格统一向量空间、BM25 + Dense 混合检索、上下文记忆、Corrective / Adaptive RAG 质量评估和 Human-in-the-loop 审核流程，并预留 GME、Milvus、Redis 等生产组件适配。

一句话总结：

> 这个项目的价值不在于 mock 模型多复杂，而在于它把一个生产级多模态 RAG 的核心工程结构拆出来了：多模态知识单元、混合检索、上下文记忆、质量评估、人工审核和生产组件适配。

