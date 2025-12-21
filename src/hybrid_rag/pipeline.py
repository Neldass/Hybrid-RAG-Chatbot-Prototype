"""Hybrid vector + graph orchestration logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from .config import AppConfig
from .graph_store import connect_graph
from .vector_store import get_embeddings, load_vector_index


@dataclass(slots=True)
class HybridAnswer:
    """Container for the generated answer and the supporting context."""

    answer: str
    vector_context: List[Document]
    graph_context: str


class HybridRAGPipeline:
    """Simple coordinator that blends FAISS + Neo4j retrieval results."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.embeddings = get_embeddings(config)
        self.vector_store = load_vector_index(
            embeddings=self.embeddings,
            persist_path=config.vector_store_path,
        )
        self.graph = connect_graph(config)
        self.graph_llm = ChatOllama(
            model=config.chat_model,
            temperature=config.temperature,
            base_url=config.ollama_base_url,
        )
        self.response_llm = ChatOllama(
            model=config.chat_model,
            temperature=config.temperature,
            base_url=config.ollama_base_url,
        )
        self.graph_chain = GraphCypherQAChain.from_llm(
            llm=self.graph_llm,
            graph=self.graph,
            return_intermediate_steps=True,
            validate_cypher=True,
            allow_dangerous_requests=True,
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You combine vector and graph context to answer technical questions."
                    " Only rely on the supplied context; admit when information is missing.",
                ),
                (
                    "human",
                    "Question: {question}\n\n"
                    "Vector context:\n{vector_context}\n\nGraph context:\n{graph_context}",
                ),
            ]
        )

    def _format_vector_context(self, docs: List[Document]) -> str:
        """Render retrieved chunks as a readable block for the LLM."""

        parts: List[str] = []
        for doc in docs:
            title = doc.metadata.get("document_name", doc.metadata.get("source", "unknown"))
            chunk_index = doc.metadata.get("chunk_index", 0)
            parts.append(f"[{title}#{chunk_index}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    def answer(self, question: str) -> HybridAnswer:
        """Answer ``question`` by blending FAISS hits with a Neo4j query."""

        vector_hits = self.vector_store.similarity_search(
            question,
            k=self.config.top_k_vectors,
        )
        graph_result: Dict[str, str] = self.graph_chain.invoke({"question": question, "query": question})
        graph_context = graph_result.get("result", "")
        intermediate = graph_result.get("intermediate_steps")
        if isinstance(intermediate, list) and intermediate:
            trimmed = intermediate[: self.config.top_k_graph]
            graph_context = f"{graph_context}\n\nCypher reasoning:\n{trimmed}"

        final_prompt = self.answer_prompt.format_messages(
            question=question,
            vector_context=self._format_vector_context(vector_hits),
            graph_context=graph_context,
        )
        response = self.response_llm.invoke(final_prompt)

        return HybridAnswer(
            answer=response.content.strip(),
            vector_context=vector_hits,
            graph_context=graph_context,
        )


def build_pipeline(config: AppConfig) -> HybridRAGPipeline:
    """Factory to keep imports contained on the CLI entry point."""

    return HybridRAGPipeline(config)
