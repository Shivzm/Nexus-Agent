from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)  # put .env keys into os.environ so LangChain can see them

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    default_model: str = "groq:llama-3.3-70b-versatile"
    chatbot_model: str = "groq:llama-3.1-8b-instant"
    react_model: str = "groq:llama-3.3-70b-versatile"

    # RAG
    rag_docs_dir: str = "data/docs"
    rag_chroma_dir: str = "data/chroma"
    rag_collection: str = "documents"
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5


settings = Settings()