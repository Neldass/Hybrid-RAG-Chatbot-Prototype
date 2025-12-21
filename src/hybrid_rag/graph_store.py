"""Neo4j helpers for the hybrid RAG prototype."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

from .config import AppConfig

UPSERT_DOCUMENT = """
MERGE (d:Document {doc_id: $doc_id})
SET d.title = $title,
    d.source_path = $source_path
RETURN d
"""

UPSERT_CHUNK = """
MATCH (d:Document {doc_id: $doc_id})
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.text = $text,
    c.chunk_index = $chunk_index
MERGE (d)-[:HAS_CHUNK]->(c)
RETURN c
"""

LINK_SEQUENCE = """
MATCH (c1:Chunk {chunk_id: $current_id})
MATCH (c2:Chunk {chunk_id: $previous_id})
MERGE (c2)-[:NEXT]->(c1)
"""


def connect_graph(config: AppConfig) -> Neo4jGraph:
    """Create a ``Neo4jGraph`` client based on the provided configuration."""

    return Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )


def bootstrap_schema(graph: Neo4jGraph) -> None:
    """Create lightweight constraints for deterministic nodes."""

    graph.query("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
    graph.query("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")


def sync_graph(graph: Neo4jGraph, documents: Iterable[Document]) -> int:
    """Populate Neo4j with the supplied document chunks.

    Returns the number of chunk nodes synced.
    """

    grouped: Dict[str, List[Document]] = defaultdict(list)
    for doc in documents:
        grouped[doc.metadata["doc_id"]].append(doc)

    total_chunks = 0
    for doc_id, chunks in grouped.items():
        for idx, chunk in enumerate(chunks):
            params = {
                "doc_id": doc_id,
                "title": chunk.metadata.get("document_name", doc_id),
                "source_path": chunk.metadata.get("source_path", doc_id),
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata.get("chunk_index", idx),
                "text": chunk.page_content,
            }
            graph.query(UPSERT_DOCUMENT, params)
            graph.query(UPSERT_CHUNK, params)
            if idx > 0:
                graph.query(
                    LINK_SEQUENCE,
                    {
                        "current_id": chunk.metadata["chunk_id"],
                        "previous_id": chunks[idx - 1].metadata["chunk_id"],
                    },
                )
            total_chunks += 1

    return total_chunks
