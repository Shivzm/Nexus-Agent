from config.settings import settings
from src.rag import reranker, store
from src.rag.indexer import index_documents
from src.rag.loader import load_documents
from src.rag import retriever
from src.rag.retriever import retrieve


def test_hybrid_retrieval_fuses_dense_and_bm25_rankings(monkeypatch):
    class FakeCollection:
        def count(self):
            return 3

        def query(self, **kwargs):
            return {
                "ids": [["dense-only", "both"]],
                "documents": [["semantic result", "overlapping result"]],
                "metadatas": [[{"source": "dense.pdf"}, {"source": "both.pdf"}]],
                "distances": [[0.1, 0.2]],
            }

    class FakeEmbedder:
        def encode(self, queries):
            class Embeddings(list):
                def tolist(self):
                    return list(self)

            return Embeddings([[0.0]])

    class FakeBM25:
        def get_scores(self, tokens):
            assert tokens == ["exact", "term"]
            return [5.0, 4.0, 0.0]

    bm25_index = (
        FakeBM25(),
        ["lexical-only", "both", "unmatched"],
        ["lexical result", "overlapping result", "unmatched result"],
        [{"source": "lexical.pdf"}, {"source": "both.pdf"}, {"source": "none.pdf"}],
    )
    monkeypatch.setattr(settings, "rag_hybrid_candidate_k", 20)
    monkeypatch.setattr(settings, "rag_rrf_k", 60)
    monkeypatch.setattr(settings, "rag_score_threshold", 0.5)
    monkeypatch.setattr(settings, "rag_retrieval_config", "hybrid")
    monkeypatch.setattr(retriever, "get_collection", FakeCollection)
    monkeypatch.setattr(retriever, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(retriever, "get_bm25_index", lambda: bm25_index)

    hits = retrieve("Exact term", top_k=3)

    assert [hit["id"] for hit in hits] == ["both", "dense-only", "lexical-only"]
    assert hits[0]["metadata"]["source"] == "both.pdf"
    assert hits[0]["rrf_score"] > hits[1]["rrf_score"]


def test_hybrid_rerank_resorts_fused_candidates(monkeypatch):
    class FakeCollection:
        def count(self):
            return 3

        def query(self, **kwargs):
            return {
                "ids": [["dense-only", "both"]],
                "documents": [["semantic result", "overlapping result"]],
                "metadatas": [[{"source": "dense.pdf"}, {"source": "both.pdf"}]],
                "distances": [[0.1, 0.2]],
            }

    class FakeEmbedder:
        def encode(self, queries):
            class Embeddings(list):
                def tolist(self):
                    return list(self)

            return Embeddings([[0.0]])

    class FakeBM25:
        def get_scores(self, tokens):
            return [5.0, 4.0, 0.0]

    class FakeReranker:
        def predict(self, pairs):
            assert [text for _, text in pairs] == [
                "overlapping result",
                "semantic result",
                "lexical result",
            ]
            return [0.1, 0.9, 0.5]

    bm25_index = (
        FakeBM25(),
        ["lexical-only", "both", "unmatched"],
        ["lexical result", "overlapping result", "unmatched result"],
        [{"source": "lexical.pdf"}, {"source": "both.pdf"}, {"source": "none.pdf"}],
    )
    monkeypatch.setattr(settings, "rag_hybrid_candidate_k", 20)
    monkeypatch.setattr(settings, "rag_reranker_candidate_k", 20)
    monkeypatch.setattr(settings, "rag_rrf_k", 60)
    monkeypatch.setattr(settings, "rag_score_threshold", 0.5)
    monkeypatch.setattr(settings, "rag_retrieval_config", "hybrid_rerank")
    monkeypatch.setattr(retriever, "get_collection", FakeCollection)
    monkeypatch.setattr(retriever, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(retriever, "get_bm25_index", lambda: bm25_index)
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeReranker())

    hits = retrieve("Exact term", top_k=2)

    assert [hit["id"] for hit in hits] == ["dense-only", "lexical-only"]
    assert [hit["rank"] for hit in hits] == [1, 2]
    assert [hit["reranker_score"] for hit in hits] == [0.9, 0.5]


def test_retrieve_uses_configured_relevance_threshold(monkeypatch):
    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "ids": [["relevant", "weak"]],
                "documents": [["relevant text", "weak text"]],
                "metadatas": [[{}, {}]],
                "distances": [[0.8, 1.2]],
            }

    class FakeEmbedder:
        def encode(self, queries):
            class Embeddings(list):
                def tolist(self):
                    return list(self)

            return Embeddings([[0.0]])

    monkeypatch.setattr(settings, "rag_score_threshold", 0.5)
    monkeypatch.setattr(retriever, "get_collection", FakeCollection)
    monkeypatch.setattr(retriever, "get_embedder", FakeEmbedder)

    hits = retrieve("query")

    assert [hit["id"] for hit in hits] == ["relevant"]


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


def test_search_documents_returns_graded_document_results(monkeypatch):
    from tools import rag

    class FakeGraph:
        def invoke(self, state):
            assert state == {"question": "Where is the answer?", "query": "Where is the answer?", "attempts": 0}
            return {
                "relevant": True,
                "hits": [{
                    "content": "The answer is in the document.",
                    "metadata": {"source_file": "paper.pdf", "page": 0},
                    "similarity_score": 0.9,
                }],
            }

    monkeypatch.setattr(rag, "_get_agentic_retrieval_graph", lambda: FakeGraph())

    assert "paper.pdf, p.1" in rag.search_documents.invoke({"query": "Where is the answer?"})


def test_search_documents_reports_web_fallback(monkeypatch):
    from tools import rag

    class FakeGraph:
        def invoke(self, state):
            return {
                "relevant": False,
                "web_results": "[https://example.test] fallback evidence",
            }

    monkeypatch.setattr(rag, "_get_agentic_retrieval_graph", lambda: FakeGraph())

    result = rag.search_documents.invoke({"query": "Question"})

    assert "Web fallback results" in result
    assert "fallback evidence" in result