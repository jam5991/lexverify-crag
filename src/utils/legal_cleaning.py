"""
Legal Citation Cleaning Utilities.

Regex-based parsing and normalization for legal citations:
- Case citations (e.g., "Smith v. Jones, 123 F.3d 456 (5th Cir. 2020)")
- Statute references (e.g., "Fla. Stat. § 768.21")
- Regulatory citations (e.g., "29 C.F.R. § 1910.134")
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedCitation:
    """A parsed legal citation."""

    raw: str
    citation_type: str  # "case", "statute", "regulation"
    normalized: str


# ── Case Citation Patterns ──
# Matches patterns like: Smith v. Jones, 123 F.3d 456 (5th Cir. 2020)
CASE_CITATION_PATTERN = re.compile(
    r"(?P<parties>[A-Z][a-zA-Z\.\s]+\s+v\.\s+[A-Z][a-zA-Z\.\s]+),?\s*"
    r"(?P<volume>\d+)\s+"
    r"(?P<reporter>[A-Z][A-Za-z\.\s]+\d*[a-z]*)\s+"
    r"(?P<page>\d+)"
    r"(?:\s*\((?P<court>[^)]+)\))?"
)

# ── Statute Patterns ──
# Matches: Fla. Stat. § 768.21, Cal. Civ. Code § 1714, 42 U.S.C. § 1983
STATUTE_PATTERN = re.compile(
    r"(?:(?P<title>\d+)\s+)?"
    r"(?P<code>[A-Z][a-zA-Z\.\s]+(?:Code|Stat\.|U\.S\.C\.))\s*"
    r"§\s*(?P<section>[\d\.]+(?:\([a-zA-Z0-9]+\))?)"
)

# ── Regulatory Patterns ──
# Matches: 29 C.F.R. § 1910.134
REGULATION_PATTERN = re.compile(
    r"(?P<title>\d+)\s+"
    r"(?P<code>C\.F\.R\.)\s*"
    r"§\s*(?P<section>[\d\.]+)"
)


def extract_citations(text: str) -> list[ParsedCitation]:
    """Extract all legal citations from text.

    Args:
        text: Legal text to parse.

    Returns:
        List of ParsedCitation objects found in the text.
    """
    citations: list[ParsedCitation] = []

    # Extract case citations
    for match in CASE_CITATION_PATTERN.finditer(text):
        raw = match.group(0).strip()
        parties = match.group("parties").strip()
        volume = match.group("volume")
        reporter = match.group("reporter").strip()
        page = match.group("page")
        court = match.group("court") or ""

        normalized = f"{parties}, {volume} {reporter} {page}"
        if court:
            normalized += f" ({court.strip()})"

        citations.append(ParsedCitation(raw=raw, citation_type="case", normalized=normalized))

    # Extract statute references
    for match in STATUTE_PATTERN.finditer(text):
        raw = match.group(0).strip()
        title = match.group("title") or ""
        code = match.group("code").strip()
        section = match.group("section")

        normalized = f"{title + ' ' if title else ''}{code} § {section}"
        citations.append(ParsedCitation(raw=raw, citation_type="statute", normalized=normalized))

    # Extract regulatory citations
    for match in REGULATION_PATTERN.finditer(text):
        raw = match.group(0).strip()
        title = match.group("title")
        section = match.group("section")

        normalized = f"{title} C.F.R. § {section}"
        citations.append(
            ParsedCitation(raw=raw, citation_type="regulation", normalized=normalized)
        )

    return citations


def normalize_citation(citation: str) -> str:
    """Normalize a single citation string.

    Removes extra whitespace, standardizes abbreviations.

    Args:
        citation: Raw citation string.

    Returns:
        Normalized citation string.
    """
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", citation.strip())

    # Standardize common abbreviations
    replacements = {
        "versus": "v.",
        " vs. ": " v. ",
        " vs ": " v. ",
        "Section": "§",
        "Sec.": "§",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized


def extract_statute_refs(text: str) -> list[str]:
    """Extract statute references from text.

    Args:
        text: Legal text to parse.

    Returns:
        List of normalized statute reference strings.
    """
    return [
        c.normalized for c in extract_citations(text) if c.citation_type == "statute"
    ]
