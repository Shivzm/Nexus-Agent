from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

from config.settings import settings


def load_documents(docs_dir: str | None = None):
    root = Path(docs_dir or settings.rag_docs_dir)
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyMuPDFLoader(str(path)).load()
        elif suffix == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for doc in loaded:
            doc.metadata.update(
                source=path.relative_to(root).as_posix(),
                source_file=path.name,
                file_type=suffix.lstrip("."),
            )
        documents.extend(loaded)
    if not documents:
        raise FileNotFoundError(f"No PDF or text documents found in {root}")
    return documents