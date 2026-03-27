"""
Pinecone Data Ingestion Script.

Loads legal documents from the processed corpus and upserts them
into the Pinecone index with embeddings and metadata.

Usage:
    python -m scripts.ingest_data
    python -m scripts.ingest_data --corpus data/processed/fl_legal_corpus.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from openai import OpenAI
from pinecone import Pinecone
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

from src.config import get_settings

app = typer.Typer(help="LexVerify Data Ingestion")
console = Console()

DEFAULT_CORPUS = Path(__file__).parent.parent / "data" / "processed" / "fl_legal_corpus.json"


def load_corpus(path: Path) -> list[dict]:
    """Load legal documents from a JSON corpus file."""
    with open(path) as f:
        return json.load(f)


def generate_embeddings(
    texts: list[str], model: str, api_key: str, dimensions: int = 1024,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts, dimensions=dimensions)
    return [item.embedding for item in response.data]


@app.command()
def ingest(
    corpus: Path = typer.Option(DEFAULT_CORPUS, help="Path to the legal corpus JSON"),
    batch_size: int = typer.Option(10, help="Batch size for embedding generation"),
    namespace: str = typer.Option("", help="Pinecone namespace (optional)"),
) -> None:
    """Ingest legal documents into Pinecone."""
    settings = get_settings()

    # Validate
    if not corpus.exists():
        console.print(f"[red]Corpus file not found: {corpus}[/red]")
        raise typer.Exit(code=1)

    if not settings.pinecone_api_key or not settings.openai_api_key:
        console.print("[red]Missing API keys. Configure your .env file.[/red]")
        raise typer.Exit(code=1)

    # Load documents
    documents = load_corpus(corpus)
    console.print(
        Panel(
            f"[bold]Corpus:[/bold] {corpus.name}\n"
            f"[bold]Documents:[/bold] {len(documents)}\n"
            f"[bold]Embedding Model:[/bold] {settings.embedding_model}\n"
            f"[bold]Pinecone Index:[/bold] {settings.pinecone_index_name}",
            title="📥 Data Ingestion",
            border_style="blue",
        )
    )

    # Initialize Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Process in batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=len(documents))

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]

            # Generate embeddings
            progress.update(task, description=f"Embedding batch {i // batch_size + 1}...")
            embeddings = generate_embeddings(
                texts, settings.embedding_model, settings.openai_api_key,
                dimensions=settings.embedding_dimensions,
            )

            # Prepare upsert vectors
            vectors = []
            for doc, embedding in zip(batch, embeddings):
                vectors.append({
                    "id": doc["id"],
                    "values": embedding,
                    "metadata": {
                        "text": doc["text"],
                        "source_uri": doc.get("source_uri", ""),
                        "jurisdiction": doc.get("jurisdiction", ""),
                        "doc_type": doc.get("doc_type", ""),
                        "area_of_law": doc.get("area_of_law", ""),
                    },
                })

            # Upsert to Pinecone
            progress.update(task, description=f"Upserting batch {i // batch_size + 1}...")
            upsert_kwargs = {"vectors": vectors}
            if namespace:
                upsert_kwargs["namespace"] = namespace
            index.upsert(**upsert_kwargs)

            progress.advance(task, len(batch))

    # Verify
    stats = index.describe_index_stats()
    console.print(
        Panel(
            f"[green]✅ Successfully ingested {len(documents)} documents![/green]\n\n"
            f"[bold]Index Stats:[/bold]\n"
            f"  Total vectors: {stats.get('total_vector_count', 'N/A')}\n"
            f"  Dimensions: {stats.get('dimension', 'N/A')}",
            title="📊 Ingestion Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
