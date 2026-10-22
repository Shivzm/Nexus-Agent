from config.settings import settings
from src.rag import store
from src.rag.indexer import index_documents
from src.rag.loader import load_documents
from src.rag.retriever import retrieve


def test_retrieve_finds_relevant_chunk(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Python is a programming language.")
    (docs / "b.txt").write_text("The Eiffel Tower is located in Paris.")

    monkeypatch.setattr(settings, "rag_docs_dir", str(docs))
    monkeypatch.setattr(settings, "rag_chroma_dir", str(tmp_path / "chroma"))
    store.get_collection.cache_clear()

    index_documents(load_documents())
    hits = retrieve("Where is the Eiffel Tower?", top_k=1)
    assert "Paris" in hits[0]["content"]