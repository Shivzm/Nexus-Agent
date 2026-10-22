import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.rag.store import get_collection, get_embedder


def index_documents(documents) -> int:
    collection, embedder = get_collection(), get_embedder()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    by_source: dict[str, list] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata["source"], []).append(chunk)

    for source, source_chunks in by_source.items():
        old_ids = collection.get(where={"source": source}, include=[])["ids"]
        texts = [c.page_content for c in source_chunks]
        ids = [
            hashlib.sha256(f"{source}\0{i}\0{t}".encode()).hexdigest()
            for i, t in enumerate(texts)
        ]
        metadatas = [
            {k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
            for c in source_chunks
        ]
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embedder.encode(texts).tolist(),
        )
        stale = set(old_ids) - set(ids)
        if stale:
            collection.delete(ids=list(stale))
    return len(chunks)