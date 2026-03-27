"""
LexVerify Evaluation Suite.

Runs RAGAS evaluation metrics against the gold-standard dataset
to benchmark LexVerify CRAG vs. baseline RAG.

Metrics:
- Faithfulness: Are generated answers grounded in the context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are the retrieved documents relevant?
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="LexVerify CRAG Evaluation Suite")
console = Console()

DATA_DIR = Path(__file__).parent.parent / "data" / "gold_standard"
RESULTS_DIR = Path(__file__).parent / "results"


def load_gold_standard(jurisdiction: str = "Florida") -> list[dict]:
    """Load gold-standard Q&A pairs for a jurisdiction.

    Args:
        jurisdiction: Target jurisdiction to load.

    Returns:
        List of Q&A pair dictionaries.
    """
    file_map = {
        "Florida": "fl_personal_injury_qa.json",
    }

    filename = file_map.get(jurisdiction)
    if not filename:
        console.print(f"[red]No gold standard data for jurisdiction: {jurisdiction}[/red]")
        raise typer.Exit(code=1)

    filepath = DATA_DIR / filename
    if not filepath.exists():
        console.print(f"[red]Gold standard file not found: {filepath}[/red]")
        raise typer.Exit(code=1)

    with open(filepath) as f:
        data = json.load(f)

    return data["pairs"]


@app.command()
def run(
    jurisdiction: str = typer.Option("Florida", help="Jurisdiction to evaluate"),
    output_format: str = typer.Option("table", help="Output format: table, json, csv"),
) -> None:
    """Run RAGAS evaluation against the gold-standard dataset."""
    console.print(
        Panel(
            f"[bold]Jurisdiction:[/bold] {jurisdiction}\n"
            f"[bold]Output:[/bold] {output_format}",
            title="📊 LexVerify Evaluation",
            border_style="blue",
        )
    )

    pairs = load_gold_standard(jurisdiction)
    console.print(f"\nLoaded [bold]{len(pairs)}[/bold] gold-standard Q&A pairs.\n")

    # Display dataset overview
    table = Table(title="Gold Standard Dataset")
    table.add_column("ID", style="dim")
    table.add_column("Query", max_width=60)
    table.add_column("Difficulty", style="cyan")
    table.add_column("Citations", style="green")

    for pair in pairs:
        table.add_row(
            pair["id"],
            pair["query"][:57] + "..." if len(pair["query"]) > 60 else pair["query"],
            pair["difficulty"],
            str(len(pair["source_citations"])),
        )

    console.print(table)

    console.print(
        "\n[yellow]⚠️  Full RAGAS evaluation requires API keys and a running pipeline.[/yellow]"
        "\n[dim]Set up your .env file and Pinecone index, then run:[/dim]"
        "\n[dim]  python -m evals.run_evals run --jurisdiction Florida[/dim]"
    )

    # Placeholder for actual RAGAS evaluation
    # When API keys are configured, this would:
    # 1. Run each query through the CRAG pipeline
    # 2. Collect (question, answer, contexts, ground_truth) tuples
    # 3. Evaluate with RAGAS metrics
    # 4. Save results to evals/results/

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save dataset overview
    overview_path = RESULTS_DIR / f"{jurisdiction.lower()}_dataset_overview.json"
    with open(overview_path, "w") as f:
        json.dump(
            {
                "jurisdiction": jurisdiction,
                "total_pairs": len(pairs),
                "difficulty_distribution": {
                    d: sum(1 for p in pairs if p["difficulty"] == d)
                    for d in set(p["difficulty"] for p in pairs)
                },
            },
            f,
            indent=2,
        )

    console.print(f"\n[green]Dataset overview saved to {overview_path}[/green]")


@app.command()
def list_datasets() -> None:
    """List available gold-standard datasets."""
    console.print("\n[bold]Available Gold-Standard Datasets:[/bold]\n")

    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        console.print(
            f"  📁 [cyan]{path.name}[/cyan] — "
            f"{data.get('jurisdiction', 'Unknown')} — "
            f"{len(data.get('pairs', []))} pairs"
        )


if __name__ == "__main__":
    app()
