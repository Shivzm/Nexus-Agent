from src.rag.indexer import index_documents
from src.rag.loader import load_documents

if __name__ == "__main__":
    print(f"Indexed {index_documents(load_documents())} chunks.")