"""RAG 问答管线：检索 → 拼 prompt → LLM 生成（带 [n] 引用）。"""
from .llm import LLMError
from .search import hybrid_search

_SYSTEM = (
    "你是一个严谨的论文检索助手。只依据「参考资料」中的论文片段回答问题；"
    "片段不足以回答时，直接回答「根据已有资料无法回答」。"
    "回答中的每个关键论断都要用 [n] 标注对应参考资料的编号。"
    "用中文回答（除非问题本身是其他语言）。"
)


def _format_block(n: int, hit) -> str:
    meta = ""
    if hit.year:
        meta += f"（{hit.year}）"
    if hit.page:
        meta += f"第{hit.page}页"
    return f"[{n}]《{hit.title}》{meta}：\n{hit.text}"


def ask(store, embedder, llm, question: str, top: int = 8, per_paper_cap: int = 3):
    """返回 (answer, sources, hits, retrieval_only)。

    - 空命中：answer="根据已有资料无法回答。"，sources=[]。
    - LLM 未配置：retrieval_only=True，answer=None，hits 为检索结果。
    - LLM 调用失败：抛 LLMError（hits 由调用方自行重取）。
    """
    hits = hybrid_search(store, embedder, question, top=top, per_paper_cap=per_paper_cap)
    if not hits:
        return "根据已有资料无法回答。", [], [], False
    if llm is None or not llm.is_configured:
        return None, [], hits, True
    blocks = "\n\n".join(_format_block(i, h) for i, h in enumerate(hits, start=1))
    user = f"问题：{question}\n\n参考资料：\n{blocks}"
    answer_text = llm.chat(_SYSTEM, user)
    sources = [
        {"n": i, "title": h.title, "year": h.year, "path": h.path, "page": h.page}
        for i, h in enumerate(hits, start=1)
    ]
    return answer_text, sources, hits, False
