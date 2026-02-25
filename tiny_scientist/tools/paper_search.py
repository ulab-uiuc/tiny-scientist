"""Improved PaperSearchTool with better reliability.

This implementation prioritizes reliability:
1. OpenAlex (primary) - Free, no API key, very stable
2. arXiv (secondary) - Good for recent ML/CS papers
3. Semantic Scholar (optional) - Requires API key for best results
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from rich import print
from smolagents import Tool


class PaperSearchTool(Tool):
    """Search academic papers with improved reliability."""

    name = "paper_search"
    description = "Search academic papers and return metadata including title, abstract, authors, and bibtex."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for finding academic papers",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (default: 10)",
            "default": 10,
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(
        self,
        engine: str = "openalex",  # openalex, arxiv, or semanticscholar
        s2_api_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.s2_api_key = s2_api_key or os.environ.get("S2_API_KEY")

    def run(self, query: str, limit: Optional[int] = 10) -> Dict[str, Dict[str, Any]]:
        """Backward-compatible run method."""
        return self.forward(query, limit)

    def forward(self, query: str, limit: Optional[int] = 10) -> Dict[str, Dict[str, Any]]:
        """Search for papers and return results."""
        limit = limit or 10

        print(f"[PaperSearch] Searching '{self.engine}' for: {query[:50]}...")

        try:
            if self.engine == "openalex":
                papers = self._search_openalex(query, limit)
            elif self.engine == "arxiv":
                papers = self._search_arxiv(query, limit)
            elif self.engine == "semanticscholar":
                papers = self._search_semanticscholar(query, limit)
            else:
                papers = self._search_openalex(query, limit)  # default
        except Exception as e:
            print(f"[PaperSearch] {self.engine} failed: {e}, trying fallback...")
            papers = self._search_with_fallback(query, limit)

        if not papers:
            print(f"[PaperSearch] No results found for: {query}")
            return {}

        # Convert to dict format
        results: Dict[str, Dict[str, Any]] = {}
        for paper in papers:
            title = paper.get("title", "Unknown")
            results[title] = paper

        print(f"[PaperSearch] Found {len(results)} papers")
        return results

    def _search_with_fallback(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Try multiple sources until one works."""
        for engine in ["openalex", "arxiv", "semanticscholar"]:
            if engine == self.engine:
                continue  # Already tried
            try:
                print(f"[PaperSearch] Trying fallback: {engine}")
                if engine == "openalex":
                    return self._search_openalex(query, limit)
                elif engine == "arxiv":
                    return self._search_arxiv(query, limit)
                elif engine == "semanticscholar":
                    return self._search_semanticscholar(query, limit)
            except Exception as e:
                print(f"[PaperSearch] {engine} also failed: {e}")
                continue
        return []

    def _search_openalex(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using OpenAlex (most reliable, no API key needed)."""
        import pyalex
        from pyalex import Works

        # Set email for polite pool (faster rate limits)
        email = os.environ.get("OPENALEX_MAIL_ADDRESS")
        if email:
            pyalex.config.email = email

        works = Works().search(query).get(per_page=limit)
        if not works:
            return []

        papers = []
        for work in works:
            paper = {
                "title": work.get("title", "Unknown"),
                "abstract": work.get("abstract") or self._get_abstract_from_inverted(work),
                "authors": self._format_openalex_authors(work),
                "year": work.get("publication_year"),
                "venue": self._get_openalex_venue(work),
                "doi": work.get("doi"),
                "url": work.get("id"),
                "citations": work.get("cited_by_count", 0),
                "bibtex": self._generate_bibtex(work),
            }
            papers.append(paper)

        return papers

    def _get_abstract_from_inverted(self, work: Dict) -> str:
        """Reconstruct abstract from inverted index."""
        inv_index = work.get("abstract_inverted_index")
        if not inv_index:
            return ""

        # Reconstruct from inverted index
        word_positions = []
        for word, positions in inv_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)

    def _format_openalex_authors(self, work: Dict) -> str:
        """Format authors from OpenAlex response."""
        authorships = work.get("authorships", [])
        if not authorships:
            return "Unknown"

        names = []
        for a in authorships[:5]:  # Limit to first 5
            author = a.get("author", {})
            name = author.get("display_name", "")
            if name:
                names.append(name)

        if len(authorships) > 5:
            return ", ".join(names) + " et al."
        return ", ".join(names) if names else "Unknown"

    def _get_openalex_venue(self, work: Dict) -> str:
        """Get venue from OpenAlex response."""
        locations = work.get("locations", [])
        for loc in locations:
            source = loc.get("source")
            if source:
                return source.get("display_name", "")
        return ""

    def _search_arxiv(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using arXiv API."""
        try:
            import arxiv
        except ImportError:
            # Fallback to manual API call
            return self._search_arxiv_manual(query, limit)

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "abstract": result.summary,
                "authors": ", ".join(a.name for a in result.authors),
                "year": result.published.year if result.published else None,
                "venue": "arXiv",
                "doi": result.doi,
                "url": result.entry_id,
                "arxiv_id": result.get_short_id(),
                "bibtex": self._generate_arxiv_bibtex(result),
            }
            papers.append(paper)

        return papers

    def _search_arxiv_manual(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search arXiv using direct API call (no arxiv package needed)."""
        import urllib.parse
        import xml.etree.ElementTree as ET
        import requests

        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={limit}&sortBy=relevance"

        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)

            authors = []
            for author_el in entry.findall("atom:author/atom:name", ns):
                if author_el.text:
                    authors.append(author_el.text.strip())

            paper = {
                "title": title_el.text.strip().replace("\n", " ") if title_el is not None else "",
                "abstract": summary_el.text.strip() if summary_el is not None else "",
                "authors": ", ".join(authors),
                "year": published_el.text[:4] if published_el is not None else None,
                "venue": "arXiv",
                "bibtex": "",  # Generate later if needed
            }
            papers.append(paper)

        return papers

    def _search_semanticscholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using Semantic Scholar API."""
        try:
            from semanticscholar import SemanticScholar
        except ImportError:
            return self._search_semanticscholar_manual(query, limit)

        sch = SemanticScholar(api_key=self.s2_api_key) if self.s2_api_key else SemanticScholar()

        results = sch.search_paper(query, limit=limit)

        papers = []
        for r in results:
            paper = {
                "title": r.title,
                "abstract": r.abstract or "",
                "authors": ", ".join(a.name for a in (r.authors or [])),
                "year": r.year,
                "venue": r.venue or "",
                "doi": r.externalIds.get("DOI") if r.externalIds else None,
                "citations": r.citationCount or 0,
                "bibtex": "",  # S2 doesn't provide bibtex directly
            }
            papers.append(paper)

        return papers

    def _search_semanticscholar_manual(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search Semantic Scholar using direct API call."""
        import requests

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,venue,citationCount,externalIds"
        }
        headers = {"User-Agent": "TinyScientist/1.0"}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"S2 API error: {response.status_code}")

        data = response.json()
        if not data.get("data"):
            return []

        papers = []
        for item in data["data"]:
            authors = item.get("authors") or []
            paper = {
                "title": item.get("title", ""),
                "abstract": item.get("abstract") or "",
                "authors": ", ".join(a.get("name", "") for a in authors),
                "year": item.get("year"),
                "venue": item.get("venue") or "",
                "citations": item.get("citationCount", 0),
                "bibtex": "",
            }
            papers.append(paper)

        return papers

    def _generate_bibtex(self, work: Dict) -> str:
        """Generate BibTeX from OpenAlex work."""
        title = work.get("title", "Unknown")
        year = work.get("publication_year", "")

        # Create citation key
        first_word = re.sub(r"[^\w]", "", title.split()[0].lower()) if title else "paper"
        key = f"{first_word}{year}"

        # Get authors
        authorships = work.get("authorships", [])
        authors = " and ".join(
            a.get("author", {}).get("display_name", "")
            for a in authorships[:10]
        )

        # Get venue
        venue = self._get_openalex_venue(work)
        doi = work.get("doi", "")

        return f"""@article{{{key},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{{year}}},
  journal = {{{venue}}},
  doi = {{{doi}}}
}}"""

    def _generate_arxiv_bibtex(self, result: Any) -> str:
        """Generate BibTeX from arxiv result."""
        title = result.title
        year = result.published.year if result.published else ""
        authors = " and ".join(a.name for a in result.authors)

        first_word = re.sub(r"[^\w]", "", title.split()[0].lower()) if title else "paper"
        key = f"{first_word}{year}"

        return f"""@article{{{key},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{{year}}},
  journal = {{arXiv preprint}},
  eprint = {{{result.get_short_id()}}}
}}"""
