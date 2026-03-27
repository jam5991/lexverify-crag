"""
Web Search Agent.

Tavily-based fallback search triggered when the CRAG evaluator scores
retrieved documents as ambiguous. Searches for contemporary legal updates
and converts results to the standard RetrievedDocument format.
"""

from __future__ import annotations

from src.config import Settings, get_settings
from src.core.retriever import RetrievedDocument


class WebSearchAgent:
    """Performs web search fallback for legal document augmentation.

    Used when the CRAG evaluator determines that retrieved documents
    are insufficient — triggers a targeted search via Tavily for
    contemporary legal updates and rulings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Tavily client."""
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.settings.tavily_api_key)
        return self._client

    async def search(
        self,
        query: str,
        jurisdiction: str | None = None,
        max_results: int = 5,
    ) -> list[RetrievedDocument]:
        """Search the web for legal information.

        Args:
            query: The legal search query.
            jurisdiction: Optional jurisdiction to focus the search.
            max_results: Maximum number of results to return.

        Returns:
            List of RetrievedDocument from web search results.
        """
        # Augment query with jurisdiction and legal context
        search_query = query
        if jurisdiction:
            search_query = f"{jurisdiction} law: {query}"

        client = self._get_client()

        # Tavily search with legal domain focus
        results = client.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=[
                "law.cornell.edu",
                "scholar.google.com",
                "casetext.com",
                "westlaw.com",
                "findlaw.com",
                "justia.com",
                "courtlistener.com",
            ],
        )

        # Convert to RetrievedDocument format
        documents = []
        for result in results.get("results", []):
            documents.append(
                RetrievedDocument(
                    content=result.get("content", ""),
                    source_uri=result.get("url", ""),
                    jurisdiction=jurisdiction or "",
                    doc_type="web_search",
                    relevance_score=result.get("score", 0.0),
                )
            )

        return documents
