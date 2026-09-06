"""
config.py
=========
All settings for the LLM microservice.
Change values here or override via .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama — local model server
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL:    str = "qwen2.5:7b"       # recommended SOTA model (or llama3.1:8b / mistral)
    EMBED_MODEL:     str = "nomic-embed-text" # dedicated embedding model — DO NOT use OLLAMA_MODEL here.
    OLLAMA_TIMEOUT:  int = 300                # seconds before timeout (CPU needs more time)

    # Temperature presets per task type
    TEMP_STRUCTURED: float = 0.0              # temperature 0.0 for deterministic JSON/charts/decisions
    TEMP_GENERAL:    float = 0.3              # temperature 0.3 for natural conversational queries

    # ChromaDB — local vector store
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    # Note: CHROMA_COLLECTION is intentionally absent.
    # Collections are per-user: vector_docs_{user_id}  (see llm/rag.py)

    # Accuracy
    CONFIDENCE_THRESHOLD: float = 0.75  # below this → flag "needs more data"
    MAX_CONTEXT_CHUNKS:   int   = 6     # RAG chunks injected per prompt

    class Config:
        env_file = ".env"


settings = Settings()

