import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class RAGConfig:
    """Configuration class for the RAG pipeline."""
    llm_provider: str = "openai"
    embedding_provider: str = "openai"

    # Model names (derived dynamically if empty)
    llm_model: str = ""
    embedding_model: str = ""

    # API Keys (loaded from env / .env by default)
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))

    # Storage settings
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_docs"

    # Chunking strategy
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval settings
    top_k: int = 5
    search_type: str = "mmr"  # "mmr" | "similarity"

    def __post_init__(self):
        if not self.llm_model:
            llm_defaults = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-4-5-20251001",
                "ollama": "llama3.2",
                "google": "gemini-2.5-flash",
            }
            self.llm_model = llm_defaults.get(self.llm_provider, "")

        if not self.embedding_model:
            embedding_defaults = {
                "openai": "text-embedding-3-small",
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
                "ollama": "nomic-embed-text",
                "google": "models/gemini-embedding-001",
            }
            self.embedding_model = embedding_defaults.get(self.embedding_provider, "")
