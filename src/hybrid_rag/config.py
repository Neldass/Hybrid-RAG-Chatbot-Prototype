"""Configuration helpers for the hybrid RAG prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    """Container for application-level settings and tunables."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    top_k_vectors: int = 4
    top_k_graph: int = 8
    chunk_size: int = 900
    chunk_overlap: int = 150
    data_dir: Path = Path("data/docs")
    vector_store_path: Path = Path("storage/vector_store")

    def ensure_artifacts(self) -> None:
        """Make sure the expected folders exist before running pipelines."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable with an optional default."""

    value = os.getenv(key, default)
    return value if value not in {None, ""} else default


def load_config(env_file: Optional[Path] = None) -> AppConfig:
    """Load environment variables and return an :class:`AppConfig`."""

    load_dotenv(dotenv_path=env_file, override=False)

    neo4j_uri = _env("NEO4J_URI")
    neo4j_username = _env("NEO4J_USERNAME")
    neo4j_password = _env("NEO4J_PASSWORD")

    missing = [
        key
        for key, value in {
            "NEO4J_URI": neo4j_uri,
            "NEO4J_USERNAME": neo4j_username,
            "NEO4J_PASSWORD": neo4j_password,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    config = AppConfig(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=_env("NEO4J_DATABASE", "neo4j"),
        embedding_model=_env("EMBEDDING_MODEL", "nomic-embed-text"),
        chat_model=_env("CHAT_MODEL", "mistral"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(_env("CHAT_TEMPERATURE", "0.0")),
        top_k_vectors=int(_env("TOP_K_VECTORS", "4")),
        top_k_graph=int(_env("TOP_K_GRAPH", "8")),
        chunk_size=int(_env("CHUNK_SIZE", "900")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "150")),
        data_dir=Path(_env("DATA_DIR", "data/docs")),
        vector_store_path=Path(_env("VECTOR_STORE_PATH", "storage/vector_store")),
    )
    config.ensure_artifacts()
    return config
