"""Command-line harness for the hybrid chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import AppConfig, load_config
from .graph_store import bootstrap_schema, connect_graph, sync_graph
from .ingest import chunk_documents, load_documents
from .pipeline import HybridAnswer, build_pipeline
from .vector_store import build_vector_index, get_embeddings

console = Console()
app = typer.Typer(help="Hybrid RAG chatbot tooling")


def _print_answer(result: HybridAnswer, *, debug: bool) -> None:
    """Pretty-print the model response with optional context."""

    console.print(Panel(result.answer, title="Assistant", subtitle="Hybrid RAG", expand=False))
    if not debug:
        return

    context_table = Table(title="Vector Context", show_lines=True)
    context_table.add_column("Chunk")
    context_table.add_column("Content")
    for doc in result.vector_context:
        chunk_label = f"{doc.metadata.get('document_name','doc')}#{doc.metadata.get('chunk_index',0)}"
        context_table.add_row(chunk_label, doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
    console.print(context_table)
    console.print(Panel(result.graph_context or "(empty)", title="Graph Context"))


def _load_config(env_file: Optional[Path]) -> AppConfig:
    """Load configuration and surface helpful error messages."""

    try:
        return load_config(env_file)
    except ValueError as exc:  # pragma: no cover - configuration guard
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def ingest(
    data_dir: Optional[Path] = typer.Option(None, help="Directory that contains technical docs."),
    env_file: Optional[Path] = typer.Option(None, help="Explicit path to a .env file."),
):
    """Ingest documentation into vector + graph stores."""

    config = _load_config(env_file)
    source_dir = data_dir or config.data_dir
    console.print(f"Reading documents from [bold]{source_dir}[/bold]")

    documents = load_documents(source_dir)
    console.print(f"Loaded {len(documents)} raw documents. Chunking...")
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    console.print(f"Prepared {len(chunks)} chunks. Building vector store...")

    embeddings = get_embeddings(config)
    build_vector_index(
        chunks,
        embeddings=embeddings,
        persist_path=config.vector_store_path,
    )
    console.print(f"Vector index saved to {config.vector_store_path}.")

    graph = connect_graph(config)
    bootstrap_schema(graph)
    synced = sync_graph(graph, chunks)
    console.print(f"Synced {synced} chunks to Neo4j at {config.neo4j_uri}.")


@app.command()
def chat(
    question: Optional[str] = typer.Option(None, "-q", "--question", help="Ask a single question and exit."),
    env_file: Optional[Path] = typer.Option(None, help="Explicit path to a .env file."),
    debug: bool = typer.Option(False, help="Print the retrieved contexts."),
):
    """Start an interactive chat session."""

    config = _load_config(env_file)
    pipeline = build_pipeline(config)

    def _ask_once(query: str) -> None:
        result = pipeline.answer(query)
        _print_answer(result, debug=debug)

    if question:
        _ask_once(question)
        raise typer.Exit()

    console.print("Type 'exit' or 'quit' to stop the chat.")
    while True:
        query = console.input("[bold blue]You[/bold blue]: ")
        if query.strip().lower() in {"exit", "quit"}:
            break
        if not query.strip():
            continue
        _ask_once(query)


if __name__ == "__main__":  # pragma: no cover
    app()
