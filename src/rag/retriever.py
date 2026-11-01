import logging
from typing import Literal

from config.settings import settings
from src.rag.lexical import get_bm25_index, tokenize
from src.rag.reranker import rerank
from src.rag.store import get_collection, get_embedder

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    search_config: Literal["dense", "hybrid", "hybrid_rerank"] | None = None,
) -> list[dict]:
    collection = get_collection()
    count = collection.count()
    if not count:
        logger.warning("Chroma collection is empty. Run: python -m scripts.ingest")
        return []

    if score_threshold is None:
        score_threshold = settings.rag_score_threshold

    if search_config is None:
        search_config = settings.rag_retrieval_config
    if search_config not in ("dense", "hybrid", "hybrid_rerank"):
        raise ValueError(f"Unsupported retrieval config: {search_config!r}")

    result_count = top_k or settings.rag_top_k
    candidate_count = (
        min(settings.rag_hybrid_candidate_k, count)
        if search_config != "dense"
        else min(result_count, count)
    )
    result = collection.query(
        query_embeddings=get_embedder().encode([query]).tolist(),
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    dense_hits = []
    for rank, (doc_id, text, meta, dist) in enumerate(
        zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        score = 1 / (1 + dist)  # same scoring as your notebook
        if score >= score_threshold:
            dense_hits.append({
                "id": doc_id,
                "content": text,
                "metadata": meta,
                "similarity_score": score,
                "distance": dist,
                "rank": rank,
            })

    if search_config == "dense":
        return dense_hits[:result_count]
    bm25, bm25_ids, bm25_documents, bm25_metadatas = get_bm25_index()
    lexical_scores = bm25.get_scores(tokenize(query))
    lexical_ranks = sorted(
        (
            (index, float(score))
            for index, score in enumerate(lexical_scores)
            if score > 0
        ),
        key=lambda item: (-item[1], bm25_ids[item[0]]),
    )[:settings.rag_hybrid_candidate_k]
    fused: dict[str, dict] = {}
    rrf_k = settings.rag_rrf_k
    for hit in dense_hits:
        rank = hit["rank"]
        fused[hit["id"]] = {
            **hit,
            "dense_score": hit["similarity_score"],
            "bm25_score": 0.0,
            "rrf_score": 1 / (rrf_k + rank),
        }
    for rank, (index, lexical_score) in enumerate(lexical_ranks, start=1):
        doc_id = bm25_ids[index]
        hit = fused.get(doc_id)
        if hit is None:
            hit = {
                "id": doc_id,
                "content": bm25_documents[index],
                "metadata": bm25_metadatas[index],
                "similarity_score": 0.0,
                "distance": None,
                "dense_score": 0.0,
                "bm25_score": lexical_score,
                "rrf_score": 0.0,
            }
            fused[doc_id] = hit
        hit["bm25_score"] = lexical_score
        hit["rrf_score"] += 1 / (rrf_k + rank)

    max_rrf_score = 2 / (rrf_k + 1)
    hits = sorted(
        fused.values(),
        key=lambda hit: (-hit["rrf_score"], hit["id"]),
    )
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
        hit["rrf_score"] /= max_rrf_score
        hit["similarity_score"] = hit["rrf_score"]
    if search_config == "hybrid_rerank":
        return rerank(query, hits[:settings.rag_reranker_candidate_k], result_count)
    return hits[:result_count]