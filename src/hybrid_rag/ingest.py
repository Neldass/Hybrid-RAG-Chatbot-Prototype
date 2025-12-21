"""Document ingestion and chunking utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_rich_documents(source_dir: Path) -> List[Document]:
    """Load HTML and PDF files that require custom loaders."""

    documents: List[Document] = []
    html_patterns = ("*.html", "*.htm")
    for pattern in html_patterns:
        for html_file in source_dir.rglob(pattern):
            loader = BSHTMLLoader(str(html_file))
            documents.extend(loader.load())

    for pdf_file in source_dir.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def load_documents(source_dir: Path) -> List[Document]:
    """Load markdown/text/PDF/HTML documents from ``source_dir``."""

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    docs: List[Document] = []
    patterns = ("**/*.md", "**/*.markdown", "**/*.txt")
    for pattern in patterns:
        loader = DirectoryLoader(
            str(source_dir),
            glob=pattern,
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
        )
        docs.extend(loader.load())

    docs.extend(_load_rich_documents(source_dir))

    if not docs:
        raise RuntimeError(
            f"No supported documents were found in {source_dir}. Add markdown, text, PDF, or HTML files and retry."
        )

    return docs


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into overlapping chunks ready for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " . ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))

    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        source_path = Path(metadata.get("source", f"doc_{idx}"))
        metadata.setdefault("doc_id", source_path.stem)
        metadata.setdefault("document_name", source_path.name)
        metadata["chunk_index"] = idx
        metadata["chunk_id"] = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        metadata["source_path"] = str(source_path)
        chunk.metadata = metadata

    return chunks
