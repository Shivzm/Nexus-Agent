from langchain_core.tools import tool

from src.rag.retriever import retrieve


@tool
def search_documents(query: str) -> str:
    """Search the user's indexed documents (PDFs and text files) and return
    relevant passages with their source file and page. Use this for questions
    about the user's own documents; use search_web for current events."""
    hits = retrieve(query)
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