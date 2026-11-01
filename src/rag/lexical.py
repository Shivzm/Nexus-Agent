"""Cached BM25 index over the chunks stored in Chroma."""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from config.settings import settings
from src.rag.store import get_collection


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


@lru_cache(maxsize=1)
def get_bm25_index() -> tuple[BM25Okapi, list[str], list[str], list[dict]]:
    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]
    if not ids or not documents or not metadatas:
        raise ValueError(f"Chroma collection {settings.rag_collection!r} has no indexed documents")

    valid = [
        (doc_id, text, metadata or {})
        for doc_id, text, metadata in zip(ids, documents, metadatas)
        if isinstance(text, str) and text.strip()
    ]
    if not valid:
        raise ValueError(f"Chroma collection {settings.rag_collection!r} has no usable text")

    tokens = [tokenize(text) for _, text, _ in valid]
    return (
        BM25Okapi(tokens),
        [doc_id for doc_id, _, _ in valid],
        [text for _, text, _ in valid],
        [metadata for _, _, metadata in valid],
    )


def clear_bm25_index() -> None:
    get_bm25_index.cache_clear()
