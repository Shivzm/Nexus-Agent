import logging

from config.settings import settings
from src.rag.store import get_collection, get_embedder

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int | None = None, score_threshold: float = 0.0) -> list[dict]:
    collection = get_collection()
    count = collection.count()
    if not count:
        logger.warning("Chroma collection is empty. Run: python -m scripts.ingest")
        return []

    k = min(top_k or settings.rag_top_k, count)
    result = collection.query(
        query_embeddings=get_embedder().encode([query]).tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for rank, (doc_id, text, meta, dist) in enumerate(
        zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        score = 1 / (1 + dist)  # same scoring as your notebook
        if score >= score_threshold:
            hits.append({"id": doc_id, "content": text, "metadata": meta,
                         "similarity_score": score, "distance": dist, "rank": rank})
    return hits