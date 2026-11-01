"""Cross-encoder reranking for retrieved document chunks."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from config.settings import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.rag_reranker_model)


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    if not hits:
        return []

    scores = get_reranker().predict([(query, hit["content"]) for hit in hits])
    if len(scores) != len(hits):
        raise ValueError(f"Reranker returned {len(scores)} scores for {len(hits)} documents")

    ranked = [
        {
            **hit,
            "reranker_score": float(score),
            "similarity_score": float(score),
        }
        for hit, score in zip(hits, scores)
    ]
    ranked.sort(key=lambda hit: (-hit["reranker_score"], -hit.get("rrf_score", 0.0), hit["id"]))
    for rank, hit in enumerate(ranked, start=1):
        hit["rank"] = rank
    return ranked[:top_k]
