from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.rag_embedding_model)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=settings.rag_chroma_dir)
    return client.get_or_create_collection(settings.rag_collection)