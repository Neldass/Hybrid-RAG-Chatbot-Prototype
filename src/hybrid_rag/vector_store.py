"""Vector store helpers for the hybrid RAG prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import AppConfig


def get_embeddings(config: AppConfig) -> OllamaEmbeddings:
    """Build the embedding model described by the configuration."""

    return OllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.ollama_base_url,
    )


def build_vector_index(
    documents: Iterable[Document],
    *,
    embeddings: OllamaEmbeddings,
    persist_path: Path,
) -> FAISS:
    """Persist document embeddings into a local FAISS vector store."""

    persist_path.mkdir(parents=True, exist_ok=True)
    store = FAISS.from_documents(list(documents), embeddings)
    store.save_local(str(persist_path))
    return store


def load_vector_index(
    *,
    embeddings: OllamaEmbeddings,
    persist_path: Path,
) -> FAISS:
    """Load an existing FAISS index from disk."""

    if not persist_path.exists():
        raise FileNotFoundError(
            "Vector store not found. Run the ingest command before chatting."
        )

    return FAISS.load_local(
        folder_path=str(persist_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
