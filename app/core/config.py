from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Required ---
    DATABASE_URL: str
    OPENROUTER_API_KEY: str

    # --- CORS ---
    ALLOW_CORS: str = "*"

    # --- Storage ---
    UPLOAD_DIR: str = "uploads"

    # --- Models (all served through OpenRouter) ---
    EMBEDDING_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    EMBEDDING_DIM: int = 2048
    GENERATION_MODEL: str = "arcee-ai/trinity-large-preview:free"
    QUERY_IMPROVER_MODEL: str = "arcee-ai/trinity-large-preview:free"
    RANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"

    # --- Ingestion tuning ---
    MAX_PDF_UPLOADS: int = 3
    MAX_CPU_WORKERS: int = 4
    EXTRACT_CHUNK_IMAGES: bool = True
    ENABLE_OCR: bool = False
    ACCELERATOR_DEVICE: str = "cpu"  # "cpu" | "cuda" | "mps"
    DOCLING_THREADS: int = 2

    # --- Retrieval tuning ---
    USE_RERANKER: bool = True
    IMPROVED_QUERIES_COUNT: int = 4
    TOP_K_INITIAL: int = 8
    TOP_K_RERANK: int = 6
    MAX_CHAT_HISTORY: int = 6

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()  # type: ignore[call-arg]
