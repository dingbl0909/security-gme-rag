from __future__ import annotations

from typing import Any, TypedDict

from app.config import get_settings
from app.evaluation import context_precision, context_recall
from app.memory import MemoryStore
from app.models import EvidenceItem, QueryResponse, RetrievalHit
from app.ragas_loop import RagasEvaluationLoop
from app.retrieval import HybridRetriever


class SecurityRagWorkflow:
    def __init__(self, retriever: HybridRetriever | None = None, memory: MemoryStore | None = None):
        self.settings = get_settings()
        self.retriever = retriever or HybridRetriever()
        self.memory = memory or MemoryStore()
        self.ragas_loop = RagasEvaluationLoop()
        self.graph = self._build_graph()

    def run(self, query: str, thread_id: str = "default-thread", image_ref: str | None = None, top_k: int | None = None) -> QueryResponse:
        state: RagState = {
            "query": query,
            "rewritten_query": "",
            "thread_id": thread_id,
            "image_ref": image_ref,
            "top_k": top_k,
            "intent": "",
            "route": "",
            "context_summary": "",
            "history_facts": [],
            "hits": [],
            "answer": "",
            "context_precision": 0.0,
            "context_recall": 0.0,
            "response_relevancy": 0.0,
            "faithfulness": 0.0,
            "needs_review": False,
            "review_reason": None,
            "review_payload": None,
            "approved": None,
            "reviewer_note": "",
            "workflow_trace": [],
            "retry_count": 0,
        }
        result = self.graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        return self._to_response(result)

    def resume(self, thread_id: str, approve: bool, reviewer_note: str = "") -> QueryResponse:
        from langgraph.types import Command

        result = self.graph.invoke(
            Command(resume={"approve": approve, "reviewer_note": reviewer_note}),
            config={"configurable": {"thread_id": thread_id}},
        )
        return self._to_response(result)

    def _build_graph(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(RagState)
        graph.add_node("classify_input", self._classify_input)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("retrieve_image", self._retrieve_image)
        graph.add_node("retrieve_text", self._retrieve_text)
        graph.add_node("grade_context", self._grade_context)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("grade_answer", self._grade_answer)
        graph.add_node("human_review", self._human_review)
        graph.add_node("save_memory", self._save_memory)

        graph.add_edge(START, "classify_input")
        graph.add_edge("classify_input", "load_memory")
        graph.add_conditional_edges(
            "load_memory",
            route_retrieval,
            {"image": "retrieve_image", "text": "retrieve_text"},
        )
        graph.add_edge("retrieve_image", "grade_context")
        graph.add_edge("retrieve_text", "grade_context")
        graph.add_conditional_edges(
            "grade_context",
            route_after_context_grade,
            {"rewrite": "rewrite_query", "generate": "generate_answer", "review": "human_review"},
        )
        graph.add_edge("rewrite_query", "retrieve_text")
        graph.add_edge("generate_answer", "grade_answer")
        graph.add_conditional_edges(
            "grade_answer",
            route_after_answer_grade,
            {"review": "human_review", "save": "save_memory"},
        )
        graph.add_edge("human_review", "save_memory")
        graph.add_edge("save_memory", END)
        return graph.compile(checkpointer=InMemorySaver())

    def _classify_input(self, state: "RagState") -> dict[str, Any]:
        intent = infer_intent(state["query"], state.get("image_ref"))
        route = "image" if state.get("image_ref") or any(word in state["query"] for word in ["图片", "截图", "画面", "图像"]) else "text"
        return {
            "intent": intent,
            "route": route,
            "workflow_trace": state["workflow_trace"] + [f"输入识别：{intent}，进入{route}检索链路。"],
        }

    def _load_memory(self, state: "RagState") -> dict[str, Any]:
        context_summary, history_facts = self.memory.load(state["thread_id"])
        return {
            "context_summary": context_summary,
            "history_facts": history_facts,
            "workflow_trace": state["workflow_trace"] + [f"历史上下文检索：读取 {len(history_facts)} 条历史事实。"],
        }

    def _retrieve_image(self, state: "RagState") -> dict[str, Any]:
        hits = self.retriever.search(state["query"], image_ref=state.get("image_ref"), top_k=state.get("top_k"))
        return {
            "hits": hits,
            "workflow_trace": state["workflow_trace"] + [f"知识库检索：图像/图文链路召回 {len(hits)} 条证据。"],
        }

    def _retrieve_text(self, state: "RagState") -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        hits = self.retriever.search(query, image_ref=None, top_k=state.get("top_k"))
        return {
            "hits": hits,
            "workflow_trace": state["workflow_trace"] + [f"知识库检索：文本混合检索召回 {len(hits)} 条证据。"],
        }

    def _grade_context(self, state: "RagState") -> dict[str, Any]:
        hits = state["hits"]
        precision = context_precision(state["query"], hits)
        recall = context_recall(state["query"], hits)
        trace = f"结果评估：context_precision={precision}，context_recall={recall}。"
        return {
            "context_precision": precision,
            "context_recall": recall,
            "workflow_trace": state["workflow_trace"] + [trace],
        }

    def _rewrite_query(self, state: "RagState") -> dict[str, Any]:
        rewritten = rewrite_query(state["query"], state["intent"])
        return {
            "rewritten_query": rewritten,
            "retry_count": state["retry_count"] + 1,
            "workflow_trace": state["workflow_trace"] + [f"Query 改写：{rewritten}"],
        }

    def _generate_answer(self, state: "RagState") -> dict[str, Any]:
        answer = compose_answer(state["query"], state["intent"], state["hits"], state["history_facts"])
        return {
            "answer": answer,
            "workflow_trace": state["workflow_trace"] + ["答案生成：基于召回证据生成可追溯回答。"],
        }

    def _grade_answer(self, state: "RagState") -> dict[str, Any]:
        metrics = self.ragas_loop.evaluate(state["query"], state["answer"], state["hits"])
        needs_review, reason = review_decision(
            state["context_precision"],
            state["context_recall"],
            metrics.response_relevancy,
            metrics.faithfulness,
            state["hits"],
        )
        self.ragas_loop.record(
            thread_id=state["thread_id"],
            query=state["query"],
            answer=state["answer"],
            hits=state["hits"],
            metrics=metrics,
            needs_review=needs_review,
            review_reason=reason,
        )
        return {
            "response_relevancy": metrics.response_relevancy,
            "faithfulness": metrics.faithfulness,
            "needs_review": needs_review,
            "review_reason": reason,
            "workflow_trace": state["workflow_trace"]
            + [
                "答案评估："
                f"response_relevancy={metrics.response_relevancy}，"
                f"faithfulness={metrics.faithfulness}，"
                f"needs_review={needs_review}。"
            ],
        }

    def _human_review(self, state: "RagState") -> dict[str, Any]:
        from langgraph.types import interrupt

        payload = {
            "thread_id": state["thread_id"],
            "query": state["query"],
            "intent": state["intent"],
            "reason": state.get("review_reason") or "证据不足或答案不可靠，需要人工审批。",
            "answer_preview": state.get("answer") or "当前尚未生成可靠答案。",
            "evidence": [to_evidence(hit).model_dump() for hit in state["hits"][:3]],
        }
        decision = interrupt(payload)
        approved = bool(decision.get("approve")) if isinstance(decision, dict) else bool(decision)
        reviewer_note = decision.get("reviewer_note", "") if isinstance(decision, dict) else ""
        if not state.get("answer"):
            answer = "当前证据不足，已进入人工审核。"
        else:
            answer = state["answer"]
        if approved:
            answer += f"\n\n人工审核：已通过。{reviewer_note}".rstrip()
        else:
            answer += f"\n\n人工审核：未通过，需要补充资料或重新检索。{reviewer_note}".rstrip()
        return {
            "answer": answer,
            "approved": approved,
            "reviewer_note": reviewer_note,
            "review_payload": payload,
            "needs_review": not approved,
            "workflow_trace": state["workflow_trace"] + ["人工审批：已从中断点恢复执行。"],
        }

    def _save_memory(self, state: "RagState") -> dict[str, Any]:
        facts = [hit.chunk.title for hit in state["hits"][:3]]
        self.memory.save_turn(state["thread_id"], state["query"], state["answer"], facts)
        return {"workflow_trace": state["workflow_trace"] + ["上下文写回：保存会话摘要和关键事实。"]}

    @staticmethod
    def _to_response(state: "RagState | dict[str, Any]") -> QueryResponse:
        interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
        if interrupts:
            interrupt_payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            return QueryResponse(
                answer="当前结果需要人工审核，请调用 /resume 继续。",
                thread_id=interrupt_payload.get("thread_id", "default-thread"),
                intent=interrupt_payload.get("intent", ""),
                evidence=[EvidenceItem(**item) for item in interrupt_payload.get("evidence", [])],
                context_precision=0.0,
                context_recall=0.0,
                response_relevancy=0.0,
                faithfulness=0.0,
                needs_review=True,
                review_reason=interrupt_payload.get("reason"),
                status="interrupted",
                workflow_trace=[],
                review_payload=interrupt_payload,
            )
        return QueryResponse(
            answer=state.get("answer", ""),
            thread_id=state.get("thread_id", "default-thread"),
            intent=state.get("intent", ""),
            evidence=[to_evidence(hit) for hit in state.get("hits", [])],
            context_summary=state.get("context_summary", ""),
            context_precision=state.get("context_precision", 0.0),
            context_recall=state.get("context_recall", 0.0),
            response_relevancy=state.get("response_relevancy", 0.0),
            faithfulness=state.get("faithfulness", 0.0),
            needs_review=state.get("needs_review", False),
            review_reason=state.get("review_reason"),
            status="completed",
            workflow_trace=state.get("workflow_trace", []),
            review_payload=state.get("review_payload"),
        )


class RagState(TypedDict):
    query: str
    rewritten_query: str
    thread_id: str
    image_ref: str | None
    top_k: int | None
    intent: str
    route: str
    context_summary: str
    history_facts: list[str]
    hits: list[RetrievalHit]
    answer: str
    context_precision: float
    context_recall: float
    response_relevancy: float
    faithfulness: float
    needs_review: bool
    review_reason: str | None
    review_payload: dict[str, Any] | None
    approved: bool | None
    reviewer_note: str
    workflow_trace: list[str]
    retry_count: int


def route_retrieval(state: RagState) -> str:
    return "image" if state["route"] == "image" else "text"


def route_after_context_grade(state: RagState) -> str:
    if not state["hits"]:
        return "review"
    if state["intent"] == "安防知识问答" and state["retry_count"] >= 1:
        return "review"
    top_score = max((hit.hybrid_score for hit in state["hits"]), default=0.0)
    weak_context = state["context_precision"] < 0.35 or state["context_recall"] < 0.2 or top_score < 0.35
    if weak_context and state["retry_count"] < 1:
        return "rewrite"
    if state["context_precision"] < 0.25 or state["context_recall"] < 0.1 or top_score < 0.35:
        return "review"
    return "generate"


def route_after_answer_grade(state: RagState) -> str:
    return "review" if state["needs_review"] else "save"


def infer_intent(query: str, image_ref: str | None = None) -> str:
    if image_ref:
        return "图文联合检索 / 现场截图问答"
    if any(word in query for word in ["图片", "截图", "画面", "图像"]):
        return "文搜图 / 图像证据检索"
    if any(word in query for word in ["告警", "误报", "布控", "越界", "入侵"]):
        return "布控告警处置"
    if any(word in query for word in ["摄像头", "设备", "离线", "接入", "心跳"]):
        return "设备接入与离线排查"
    if any(word in query for word in ["部署", "接口", "服务", "Milvus", "Redis", "OCR"]):
        return "私有化部署排障"
    return "安防知识问答"


def rewrite_query(query: str, intent: str) -> str:
    if "摄像头" in query or "离线" in query:
        return f"{query} 设备离线 心跳 GB28181 RTSP 排查 SOP"
    if "告警" in query or "误报" in query:
        return f"{query} 告警误报 布控区域 阈值 现场截图 SOP"
    if "部署" in query or "接口" in query:
        return f"{query} 私有化部署 接口异常 Milvus Redis OCR 配置"
    return f"{query} {intent} 安防知识库 处置建议"


def compose_answer(query: str, intent: str, hits: list[RetrievalHit], history_facts: list[str]) -> str:
    lines = [f"问题类型：{intent}", "结论："]
    if not hits:
        lines.append("当前知识库没有召回足够证据，建议进入人工审核或补充资料。")
        return "\n".join(lines)
    lines.append("已基于混合检索召回相关证据，建议按证据中的 SOP 逐项排查。")
    if history_facts:
        lines.append(f"已参考历史上下文：{', '.join(history_facts[:3])}。")
    lines.append("")
    lines.append("关键证据：")
    for index, hit in enumerate(hits[:3], start=1):
        snippet = hit.chunk.content.replace("\n", " ")[:120]
        lines.append(f"{index}. {hit.chunk.title}（{hit.chunk.modality}，score={hit.hybrid_score}）：{snippet}")
    lines.append("")
    lines.append("建议：")
    lines.append("1. 先确认召回证据是否覆盖当前现象。")
    lines.append("2. 根据 SOP 检查设备、告警规则或部署配置。")
    lines.append("3. 若证据不足或涉及生产变更，进入人工审核。")
    return "\n".join(lines)


def review_decision(
    precision: float,
    recall: float,
    relevancy: float,
    faithful: float,
    hits: list[RetrievalHit],
) -> tuple[bool, str | None]:
    if not hits:
        return True, "未召回证据。"
    top_score = max((hit.hybrid_score for hit in hits), default=0.0)
    if top_score < 0.35:
        return True, "最高混合检索分数偏低，证据相关性不足。"
    if precision < 0.35 or recall < 0.2:
        return True, "检索证据覆盖不足，需要 Query 改写或人工补充资料。"
    if relevancy < 0.2:
        return True, "回答与用户问题相关性偏低，需要人工审核。"
    if faithful < 0.25:
        return True, "回答与证据一致性偏低，需要人工审核。"
    return False, None


def to_evidence(hit: RetrievalHit) -> EvidenceItem:
    return EvidenceItem(
        title=hit.chunk.title,
        source=hit.chunk.source,
        modality=hit.chunk.modality,
        snippet=hit.chunk.content.replace("\n", " ")[:180],
        image_ref=hit.chunk.image_ref,
        dense_score=hit.dense_score,
        sparse_score=hit.sparse_score,
        hybrid_score=hit.hybrid_score,
    )

