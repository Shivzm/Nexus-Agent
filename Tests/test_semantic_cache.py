import sqlite3
from pathlib import Path

import pytest

from src.rag.semantic_cache import SemanticResponseCache, clear_semantic_cache


EMBEDDINGS = {
    "question about retrieval": [1.0, 0.0],
    "retrieval question": [0.99, 0.1],
    "unrelated arithmetic": [0.0, 1.0],
}


def make_cache(path: Path, threshold=0.95, namespace="model-v1", max_entries=10):
    return SemanticResponseCache(
        path,
        embed=lambda text: EMBEDDINGS[text],
        namespace=namespace,
        threshold=threshold,
        max_entries=max_entries,
    )


def test_cache_returns_semantically_similar_responses_and_tracks_hits(tmp_path):
    cache = make_cache(tmp_path / "semantic.sqlite")
    cache.put("question about retrieval", {"answer": "cached", "sources": []})

    hit = cache.get("retrieval question")

    assert hit["answer"] == "cached"
    assert hit["cache_similarity"] > 0.95
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_misses_for_weak_similarity_and_persists_across_instances(tmp_path):
    path = tmp_path / "semantic.sqlite"
    cache = make_cache(path)
    cache.put("question about retrieval", {"answer": "cached", "sources": []})

    assert cache.get("unrelated arithmetic") is None
    assert make_cache(path).get("question about retrieval")["answer"] == "cached"


def test_cache_namespaces_prevent_cross_model_reuse(tmp_path):
    path = tmp_path / "semantic.sqlite"
    make_cache(path, namespace="model-a").put("question about retrieval", {"answer": "a"})

    assert make_cache(path, namespace="model-b").get("question about retrieval") is None


def test_cache_caps_entries_per_namespace(tmp_path):
    path = tmp_path / "semantic.sqlite"
    cache = make_cache(path, threshold=1.0, max_entries=1)
    cache.put("question about retrieval", {"answer": "old"})
    cache.put("retrieval question", {"answer": "new"})

    with sqlite3.connect(path) as db:
        count = db.execute(
            "SELECT count(*) FROM semantic_responses WHERE namespace = ?",
            ("model-v1",),
        ).fetchone()[0]
    assert count == 1


def test_cache_rejects_bad_embedding_and_configuration(tmp_path):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        make_cache(tmp_path / "bad.sqlite", threshold=0)

    cache = SemanticResponseCache(
        tmp_path / "bad-vector.sqlite",
        embed=lambda _: [0.0, 0.0],
        namespace="test",
    )
    with pytest.raises(ValueError, match="zero vector"):
        cache.get("query")


def test_clear_semantic_cache_invalidates_all_cached_answers(tmp_path):
    path = tmp_path / "semantic.sqlite"
    cache = make_cache(path)
    cache.put("question about retrieval", {"answer": "cached"})

    clear_semantic_cache(path)

    assert make_cache(path).get("question about retrieval") is None
