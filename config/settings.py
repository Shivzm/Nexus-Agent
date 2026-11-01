from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)  # put .env keys into os.environ so LangChain can see them

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    default_model: str = "groq:openai/gpt-oss-120b"
    chatbot_model: str = "groq:openai/gpt-oss-20b"
    react_model: str = "groq:openai/gpt-oss-120b"

    # RAG
    rag_docs_dir: str = "data/docs"
    rag_chroma_dir: str = "data/chroma"
    rag_collection: str = "documents"
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    # Initial cutoff for the L2-derived similarity score; calibrate with reviewed eval data.
    rag_score_threshold: float = 0.5
    rag_retrieval_config: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"
    rag_hybrid_candidate_k: int = 20
    rag_rrf_k: int = 60
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_reranker_candidate_k: int = 20


settings = Settings()