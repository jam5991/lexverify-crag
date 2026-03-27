"""
LexVerify CLI — Main entry point for the CRAG pipeline.

Assembles the corrective RAG pipeline using LangGraph and provides
a CLI interface for running legal queries.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import get_settings
from src.utils.logger import setup_logging

app = typer.Typer(
    name="lexverify",
    help="LexVerify: Corrective RAG for Legal Citation Integrity",
    add_completion=False,
)
console = Console()


# ── LangGraph Pipeline State ──

from typing import Any

from pydantic import BaseModel, Field

from src.core.evaluator import CRAGAction, EvaluationResult
from src.core.retriever import RetrievedDocument
from src.core.router import RoutingResult


class PipelineState(BaseModel):
    """State passed through the LangGraph CRAG pipeline."""

    query: str = ""
    jurisdiction_hint: str | None = None

    # After routing
    routing: RoutingResult | None = None

    # After retrieval
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)

    # After evaluation
    evaluation: EvaluationResult | None = None

    # After augmentation (web search)
    augmented_documents: list[RetrievedDocument] = Field(default_factory=list)

    # After generation
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)

    # After hallucination grading
    is_grounded: bool = False
    grounded_ratio: float = 0.0
    ungrounded_claims: list[str] = Field(default_factory=list)


# ── Pipeline Node Functions ──


async def route_query(state: PipelineState) -> dict:
    """Route the query to the appropriate jurisdiction."""
    from src.core.router import JurisdictionalRouter

    router = JurisdictionalRouter()
    result = await router.route(state.query, state.jurisdiction_hint)
    return {"routing": result}


async def retrieve_documents(state: PipelineState) -> dict:
    """Retrieve relevant legal documents from Pinecone."""
    from src.core.retriever import LegalRetriever

    retriever = LegalRetriever()
    jurisdiction = state.routing.state if state.routing else None
    docs = await retriever.retrieve(state.query, jurisdiction=jurisdiction)
    return {"retrieved_documents": docs}


async def evaluate_documents(state: PipelineState) -> dict:
    """Evaluate retrieved documents with the CRAG critic."""
    from src.core.evaluator import CRAGEvaluator

    evaluator = CRAGEvaluator()
    jurisdiction = state.routing.state if state.routing else ""
    result = await evaluator.evaluate(
        state.query, state.retrieved_documents, jurisdiction=jurisdiction or ""
    )
    return {"evaluation": result}


async def augment_with_search(state: PipelineState) -> dict:
    """Augment documents via web search fallback."""
    from src.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    jurisdiction = state.routing.state if state.routing else None
    docs = await agent.search(state.query, jurisdiction=jurisdiction)
    return {"augmented_documents": docs}


async def generate_response(state: PipelineState) -> dict:
    """Generate the final response with citations."""
    from src.core.generator import LegalGenerator

    generator = LegalGenerator()

    # Use verified docs from evaluation + any augmented docs
    verified_docs = []
    if state.evaluation and state.evaluation.verified_documents:
        verified_docs = [
            state.retrieved_documents[i]
            for i in state.evaluation.verified_documents
            if i < len(state.retrieved_documents)
        ]

    # Add augmented documents
    all_docs = verified_docs + state.augmented_documents

    jurisdiction = state.routing.state if state.routing else ""
    response = await generator.generate(state.query, all_docs, jurisdiction=jurisdiction or "")

    return {
        "answer": response.answer,
        "citations": [c.model_dump() for c in response.citations],
    }


async def grade_response(state: PipelineState) -> dict:
    """Grade the response for hallucination."""
    from src.agents.grader import HallucinationGrader

    grader = HallucinationGrader()

    # Combine all source documents
    all_docs = state.retrieved_documents + state.augmented_documents
    result = await grader.grade(state.answer, all_docs)

    return {
        "is_grounded": result.is_grounded,
        "grounded_ratio": result.grounded_ratio,
        "ungrounded_claims": result.ungrounded_claims,
    }


def should_augment(state: PipelineState) -> str:
    """Conditional edge: decide whether to augment or generate."""
    if state.evaluation is None:
        return "augment"
    if state.evaluation.action == CRAGAction.AUGMENT:
        return "augment"
    if state.evaluation.action == CRAGAction.REINDEX:
        return "augment"  # Also try web search for reindex cases
    return "generate"


def build_pipeline():
    """Build the LangGraph CRAG pipeline."""
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("evaluate", evaluate_documents)
    workflow.add_node("augment", augment_with_search)
    workflow.add_node("generate", generate_response)
    workflow.add_node("grade", grade_response)

    # Define edges
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "evaluate")

    # Conditional: evaluate → augment or generate
    workflow.add_conditional_edges(
        "evaluate",
        should_augment,
        {"augment": "augment", "generate": "generate"},
    )
    workflow.add_edge("augment", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", END)

    return workflow.compile()


# ── CLI Commands ──


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Legal question to answer")],
    jurisdiction: Annotated[
        str | None,
        typer.Option("--jurisdiction", "-j", help="Jurisdiction hint (e.g., 'Florida')"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """Run a legal query through the LexVerify CRAG pipeline."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    settings = get_settings()

    # Validate API keys
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
        console.print(
            Panel(
                "[red]Missing OpenAI API key.[/red]\n\n"
                "Copy .env.example to .env and fill in your API keys:\n"
                "[dim]cp .env.example .env[/dim]",
                title="⚠️  Configuration Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Query:[/bold cyan] {question}\n"
            f"[bold cyan]Jurisdiction:[/bold cyan] {jurisdiction or 'Auto-detect'}",
            title="🔍 LexVerify CRAG Pipeline",
            border_style="cyan",
        )
    )

    async def _run():
        pipeline = build_pipeline()
        initial_state = PipelineState(query=question, jurisdiction_hint=jurisdiction)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running CRAG pipeline...", total=None)

            result = await pipeline.ainvoke(initial_state)

            progress.update(task, description="Complete!")

        return result

    result = asyncio.run(_run())

    # Display results
    if result.get("answer"):
        console.print()
        console.print(Panel(Markdown(result["answer"]), title="📋 Legal Response", border_style="green"))

        if result.get("citations"):
            console.print("\n[bold]📚 Citations:[/bold]")
            for i, cit in enumerate(result["citations"], 1):
                console.print(f"  [{i}] {cit.get('source_uri', 'N/A')} — {cit.get('claim', '')[:80]}")

        if result.get("is_grounded") is not None:
            grounded = result["is_grounded"]
            ratio = result.get("grounded_ratio", 0)
            icon = "✅" if grounded else "⚠️"
            color = "green" if grounded else "yellow"
            console.print(
                f"\n{icon} [bold {color}]Grounding Score:[/bold {color}] "
                f"{ratio:.0%} of claims verified"
            )

            if result.get("ungrounded_claims"):
                console.print("\n[bold yellow]⚠️  Ungrounded Claims:[/bold yellow]")
                for claim in result["ungrounded_claims"]:
                    console.print(f"  • {claim}")
    else:
        console.print("[yellow]No response generated.[/yellow]")


@app.command()
def info() -> None:
    """Display LexVerify configuration info."""
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]Generator Model:[/bold] {settings.generator_model}\n"
            f"[bold]Evaluator Model:[/bold] {settings.evaluator_model}\n"
            f"[bold]Embedding Model:[/bold] {settings.embedding_model}\n"
            f"[bold]Top-K Retrieval:[/bold] {settings.top_k}\n"
            f"[bold]Confidence High:[/bold] {settings.confidence_threshold_high}\n"
            f"[bold]Confidence Low:[/bold] {settings.confidence_threshold_low}\n"
            f"[bold]Pinecone Index:[/bold] {settings.pinecone_index_name}\n"
            f"[bold]OpenAI Key:[/bold] {'✅ Set' if settings.openai_api_key else '❌ Missing'}\n"
            f"[bold]Tavily Key:[/bold] {'✅ Set' if settings.tavily_api_key else '❌ Missing'}\n"
            f"[bold]Pinecone Key:[/bold] {'✅ Set' if settings.pinecone_api_key else '❌ Missing'}",
            title="⚙️  LexVerify Configuration",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    app()
