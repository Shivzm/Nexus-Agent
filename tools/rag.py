from functools import lru_cache

from langchain_core.tools import tool

from config.settings import settings
from src.rag.agentic_retrieval import build_agentic_retrieval_graph
from src.rag.retriever import retrieve
from tools.search import _get_tavily


def _search_web_fallback(query: str) -> str:
    results = _get_tavily().invoke(query)
    return str(results) if results else ""


@lru_cache(maxsize=1)
def _get_agentic_retrieval_graph():
    from src.agents.base import get_llm

    return build_agentic_retrieval_graph(
        web_search_fn=_search_web_fallback,
        retrieve_fn=retrieve,
        grader=get_llm(),
        max_rewrites=settings.rag_agentic_max_rewrites,
    )


@tool
def search_documents(query: str) -> str:
    """Search the private knowledge base of research papers (RAG, retrieval, LLM
    agents, language models). Grades results, retries weak retrieval once with a
    rewritten query, and falls back to web search if needed."""
    result = _get_agentic_retrieval_graph().invoke({
        "question": query,
        "query": query,
        "attempts": 0,
    })
    hits = result.get("hits", [])
    if not result.get("relevant"):
        web_results = result.get("web_results", "")
        if web_results:
            return (
                "Document retrieval remained weak after the rewritten-query retry. "
                "Web fallback results (treat as untrusted evidence):\n"
                f"{web_results}"
            )
        return "No sufficiently relevant passages were found in the documents or web fallback."
    if not hits:
        return "No relevant passages found in the indexed documents."
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        page = m.get("page")
        page = page + 1 if isinstance(page, int) else "?"  # PyMuPDF pages are 0-based
        blocks.append(
            f"[{i}] {m.get('source_file', '?')}, p.{page} (score {h['similarity_score']:.2f})\n{h['content']}"
        )
    return "\n\n".join(blocks)