from __future__ import annotations

import abc
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import requests
import toml
from fastmcp import FastMCP  # type: ignore
from rich import print

import tiny_scientist
from tiny_scientist.budget_checker import BudgetChecker
from tiny_scientist.utils.error_handler import api_calling_error_exponential_backoff

PACKAGE_ROOT = Path(tiny_scientist.__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config.toml"
config = toml.load(CONFIG_PATH) if CONFIG_PATH.exists() else {"core": {}}


class BaseTool(abc.ABC):
    def __init__(self, cost_tracker: Optional[BudgetChecker] = None) -> None:
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.github_token = config["core"].get("github_token", None)

    @abc.abstractmethod
    def run(
        self, query: str, result_limit: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        pass


class PaperSearchTool(BaseTool):
    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        engine: Optional[str] = None,
        disable_fallback: bool = False,
    ) -> None:
        super().__init__()
        raw_key = (
            s2_api_key
            or os.environ.get("S2_API_KEY")
            or config["core"].get("s2_api_key")
        )
        self.s2_api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key
        self.disable_fallback = disable_fallback

        # Engine selection priority: explicit param > config file > S2 key present -> semanticscholar > openalex
        configured_engine = config["core"].get("engine")
        if engine:
            self.engine = engine
        elif configured_engine:
            self.engine = configured_engine
        elif self.s2_api_key:
            self.engine = "semanticscholar"
        else:
            self.engine = "openalex"

        # Print configuration info
        print(f"[INFO] Primary search engine: {self.engine}")
        if self.disable_fallback:
            print("[INFO] Fallback to alternative search engine is DISABLED")

        if self.engine == "semanticscholar":
            if not self.disable_fallback:
                print(
                    "[INFO] Will fallback to arXiv → OpenAlex if Semantic Scholar fails"
                )
            if not self.s2_api_key:
                print("[INFO] No S2_API_KEY, rate limits will be stricter")
        elif self.engine == "openalex":
            mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
            if mail:
                print(f"[INFO] OpenAlex email configured: {mail}")
            else:
                print(
                    "[INFO] Recommend setting OPENALEX_MAIL_ADDRESS environment variable for better API access"
                )

        print(
            "[INFO] If you encounter search issues, OpenAlex is usually more stable than Semantic Scholar"
        )

    def run(self, query: str, result_limit: int = 10) -> Dict[str, Dict[str, str]]:
        results = {}
        print(f"[PaperSearchTool] Searching for: {query}")
        papers = self.search_for_papers(query, result_limit=result_limit)

        if papers:
            print(f"[PaperSearchTool] Found {len(papers)} papers")
            for i, paper in enumerate(papers):
                paper_title = paper.get("title", "Unknown Title")
                print(f"[PaperSearchTool] Processing paper {i+1}: {paper_title}")

                # Strategy: Prefer OpenAlex for bibtex (more reliable formatting)
                paper_data = {
                    "title": paper_title,
                    "abstract": paper.get("abstract") or "",  # Handle None
                    "authors": paper.get("authors") or "",
                    "venue": paper.get("venue") or "",
                    "year": paper.get("year") or "",
                    "citationCount": paper.get("citationCount", 0),
                    "concepts": paper.get("concepts", []),
                    "bibtex": "",
                }

                # Priority 1: Try OpenAlex for bibtex (best formatting)
                # First try if we have openalex_id
                if "openalex_id" in paper and paper.get("openalex_id"):
                    openalex_id = paper["openalex_id"]
                    bibtex = self._fetch_bibtex_from_openalex(openalex_id)
                    if bibtex:
                        paper_data["bibtex"] = bibtex
                        print("[PaperSearchTool] ✅ Got bibtex from OpenAlex (via ID)")

                # If no bibtex yet, try searching OpenAlex by title (OPTIMIZED: 1 API call)
                """
                if not paper_data["bibtex"]:
                    try:
                        print("[PaperSearchTool] No OpenAlex ID, searching by title for bibtex...")
                        # OPTIMIZATION: Directly fetch bibtex in one API call using title
                        bibtex = self._fetch_bibtex_by_title(paper_title)
                        if bibtex:
                            paper_data["bibtex"] = bibtex
                            print(
                                "[PaperSearchTool] ✅ Got bibtex from OpenAlex (via optimized title search)"
                            )
                    except Exception as e:
                        print(f"[PaperSearchTool] OpenAlex title search failed: {e}")
                """

                if not paper_data["bibtex"]:
                    bibtex = self._generate_bibtex_from_metadata(paper)
                    if bibtex:
                        paper_data["bibtex"] = bibtex
                        print(
                            "[PaperSearchTool] ⚠️ Generated bibtex from metadata (may have formatting issues)"
                        )
                    else:
                        print("[PaperSearchTool] ❌ No bibtex available")
                        continue  # Skip papers without bibtex

                # Try to enrich with arXiv if abstract is missing or too short
                abstract_text = paper_data["abstract"] or ""
                if len(abstract_text) < 100:
                    try:
                        print(
                            f"[PaperSearchTool] Abstract too short ({len(abstract_text)} chars), trying arXiv..."
                        )
                        arxiv_abstract = self._fetch_abstract_from_arxiv(paper_title)
                        if arxiv_abstract and len(arxiv_abstract) > len(abstract_text):
                            paper_data["abstract"] = arxiv_abstract
                            print(
                                f"[PaperSearchTool] ✅ Enriched with arXiv abstract ({len(arxiv_abstract)} chars)"
                            )
                        else:
                            print(
                                "[PaperSearchTool] ⚠️ arXiv enrichment failed or no improvement"
                            )
                    except Exception as e:
                        print(f"[PaperSearchTool] arXiv enrichment failed: {e}")

                results[paper_title] = paper_data
                abstract_len = len(paper_data.get("abstract", ""))
                print(f"[PaperSearchTool] 📝 Final abstract length: {abstract_len}")
        else:
            print(f"[PaperSearchTool] ❌ No papers found for query: {query}")

        print(f"[PaperSearchTool] Final results: {len(results)} papers with bibtex")
        self.cost_tracker.report()
        return results

    def search_for_papers(
        self, query: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        if not query:
            return None

        # Engine preference with graceful fallback
        if self.engine == "semanticscholar":
            print(
                f"(semantic scholar API calling) Searching for papers with query: {query}"
            )
            try:
                result = self._search_semanticscholar(query, result_limit)
                if result:
                    return result
                else:
                    if self.disable_fallback:
                        print(
                            "[WARNING] Semantic Scholar returned no results. Fallback is disabled."
                        )
                        return None
                    print(
                        "[INFO] Semantic Scholar returned no results, trying arXiv..."
                    )
            except Exception as e:
                if self.disable_fallback:
                    print(
                        f"[ERROR] Semantic Scholar failed: {e}. Fallback is disabled."
                    )
                    return None
                print(
                    f"[WARNING] Semantic Scholar failed: {e}, trying arXiv as fallback..."
                )

            # Fallback to arXiv (only if not disabled)
            try:
                print(f"(arXiv API calling) Fallback search with query: {query}")
                arxiv_result = self._search_arxiv(query, result_limit)
                if arxiv_result:
                    return arxiv_result
                # If arXiv also fails, try OpenAlex as last resort
                print(
                    "[INFO] arXiv also returned no results, trying OpenAlex as last resort..."
                )
                return self._search_openalex(query, result_limit)
            except Exception as e:
                print(
                    f"[ERROR] All search engines failed (Semantic Scholar → arXiv → OpenAlex): {e}"
                )
                return None

        elif self.engine == "openalex":
            print(f"(openalex API calling) Searching for papers with query: {query}")
            try:
                result = self._search_openalex(query, result_limit)
                if result:
                    return result
                else:
                    if self.disable_fallback:
                        print(
                            "[WARNING] OpenAlex returned no results. Fallback is disabled."
                        )
                        return None
                    print(
                        "[WARNING] OpenAlex returned no results, trying Semantic Scholar..."
                    )
            except Exception as e:
                if self.disable_fallback:
                    print(f"[ERROR] OpenAlex failed: {e}. Fallback is disabled.")
                    return None
                print(
                    f"[WARNING] OpenAlex failed: {e}, trying Semantic Scholar as fallback..."
                )

            # Fallback to Semantic Scholar (only if not disabled)
            try:
                print(
                    f"(semantic scholar API calling) Fallback search with query: {query}"
                )
                return self._search_semanticscholar(query, result_limit)
            except Exception as e:
                print(f"[ERROR] Both OpenAlex and Semantic Scholar failed: {e}")
                return None
        else:
            raise NotImplementedError(f"{self.engine=} not supported!")

    @api_calling_error_exponential_backoff(retries=3, base_wait_time=2)
    def _search_semanticscholar(
        self, query: str, result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, str | int] = {
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,paperId",
        }

        # Set comprehensive headers
        headers = {
            "User-Agent": "TinyScientist/1.0 (https://github.com/ulab-uiuc/tiny-scientiest)",
            "Accept": "application/json",
        }
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        try:
            print(f"[Semantic Scholar] Searching for: {query[:100]}...")
            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params=params,
                timeout=30,
            )
            print(f"[Semantic Scholar] Status code: {rsp.status_code}")
            rsp.raise_for_status()

            results = rsp.json()
            if not results.get("total"):
                print(f"[Semantic Scholar] No results found for query: {query}")
                return None

            print(f"[Semantic Scholar] Found {results.get('total')} papers")
            time.sleep(2.0)  # Add delay to avoid rate limiting
            return cast(Optional[List[Dict[str, Any]]], results.get("data"))

        except requests.exceptions.HTTPError as e:
            if rsp.status_code == 403:
                print("[Semantic Scholar] 403 Forbidden - API access denied")
                print("[INFO] This could be due to:")
                print("  1. Missing or invalid API key")
                print("  2. Rate limiting")
                print("  3. IP restrictions")
                if not self.s2_api_key:
                    print(
                        "[SUGGESTION] Get a free API key at: https://www.semanticscholar.org/product/api"
                    )
                # For 403 errors, don't retry, return None to allow fallback
                print(
                    "[Semantic Scholar] Skipping retries for 403 error, will try fallback engine"
                )
                return None
            elif rsp.status_code == 429:
                print(
                    "[Semantic Scholar] 429 Rate Limited - will retry with exponential backoff"
                )
                # For 429, let decorator handle retries
                raise
            else:
                print(f"[Semantic Scholar] HTTP Error {rsp.status_code}: {e}")
                raise
        except requests.exceptions.RequestException as e:
            print(f"[Semantic Scholar] Request failed: {e}")
            raise
        except Exception as e:
            print(f"[Semantic Scholar] Unexpected error: {e}")
            raise

    def _search_openalex(
        self, query: str, result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        import pyalex
        from pyalex import Works

        mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
        if mail:
            pyalex.config.email = mail
        else:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better API access")

        works = Works().search(query).get(per_page=result_limit)
        if not works:
            return None

        return [self._extract_work_info(work) for work in works]

    def _fetch_bibtex_by_title(self, title: str) -> Optional[str]:
        """
        SUPER OPTIMIZED: Fetch BibTeX by title in truly ONE API call!
        Uses OpenAlex filter search + format parameter to get bibtex directly
        """
        try:
            import requests

            # STEP 1: Quick search to get work ID (lightweight, JSON response)
            search_url = "https://api.openalex.org/works"
            params = {
                "filter": f"title.search:{title}",
                "per_page": 1,
                "select": "id",  # Only get ID field for speed
            }

            headers = {}
            mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
            if mail:
                headers["User-Agent"] = f"TinyScientist (mailto:{mail})"

            response = requests.get(
                search_url, params=params, headers=headers, timeout=10
            )
            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get("results", [])
            if not results or not results[0].get("id"):
                return None

            work_id = results[0]["id"]

            # STEP 2: Fetch bibtex using the ID (still 2 calls but both are optimized)
            return self._fetch_bibtex_from_openalex(work_id)

        except Exception as e:
            print(f"[WARNING] Failed to fetch bibtex by title from OpenAlex: {e}")
            return None

    def _fetch_bibtex_from_openalex(self, work_id: str) -> Optional[str]:
        """Fetch BibTeX from OpenAlex by work ID (OpenAlex ID)"""
        try:
            import requests

            # OpenAlex provides direct bibtex endpoint
            # work_id should be like "W2741809807" or full URL
            if work_id.startswith("http"):
                bibtex_url = work_id.replace(
                    "https://openalex.org/", "https://api.openalex.org/"
                )
            else:
                bibtex_url = f"https://api.openalex.org/works/{work_id}"

            headers = {"Accept": "application/x-bibtex"}
            mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
            if mail:
                headers["User-Agent"] = f"TinyScientist (mailto:{mail})"

            response = requests.get(bibtex_url, headers=headers, timeout=10)
            if response.status_code == 200 and response.text:
                return response.text.strip()
            return None
        except Exception as e:
            print(f"[WARNING] Failed to fetch bibtex from OpenAlex: {e}")
            return None

    def _generate_bibtex_from_metadata(self, paper: Dict[str, Any]) -> str:
        """Generate BibTeX entry from paper metadata with proper author formatting"""
        try:
            title = paper.get("title", "Unknown Title")
            authors_raw = paper.get("authors", "Unknown Author")
            venue = paper.get("venue", "Unknown Venue")
            year = paper.get("year", "Unknown")

            # Format authors properly
            if isinstance(authors_raw, list):
                # Handle list of dicts (Semantic Scholar format)
                if authors_raw and isinstance(authors_raw[0], dict):
                    author_names = [
                        a.get("name", "") for a in authors_raw if a.get("name")
                    ]
                    authors = (
                        " and ".join(author_names) if author_names else "Unknown Author"
                    )
                # Handle list of strings
                elif authors_raw and isinstance(authors_raw[0], str):
                    authors = " and ".join(authors_raw)
                else:
                    authors = "Unknown Author"
            elif isinstance(authors_raw, str):
                # If it's already a string and looks like Python dict, try to extract names
                if "[{" in authors_raw or "authorId" in authors_raw:
                    try:
                        import ast

                        parsed = ast.literal_eval(authors_raw)
                        if isinstance(parsed, list) and parsed:
                            author_names = [
                                a.get("name", "")
                                for a in parsed
                                if isinstance(a, dict) and a.get("name")
                            ]
                            authors = (
                                " and ".join(author_names)
                                if author_names
                                else "Unknown Author"
                            )
                        else:
                            authors = "Unknown Author"
                    except Exception:
                        authors = "Unknown Author"
                else:
                    authors = authors_raw
            else:
                authors = "Unknown Author"

            # Generate bibtex key (clean special characters)
            import re

            clean_title = re.sub(r"[^\w\s]", "", title)
            first_word = clean_title.split()[0] if clean_title.split() else "paper"
            bibtex_key = f"{first_word.lower()}{year}"

            # Build BibTeX entry
            bibtex = f"""@article{{{bibtex_key},
    title={{{title}}},
    author={{{authors}}},
    journal={{{venue}}},
    year={{{year}}}
}}"""
            return bibtex

        except Exception as e:
            print(f"[ERROR] Failed to generate bibtex: {e}")
            return ""

    @api_calling_error_exponential_backoff(retries=1, base_wait_time=1)
    def fetch_bibtex(self, paper_id: str) -> Any:
        # Set comprehensive headers
        headers = {
            "User-Agent": "TinyScientist/1.0 (https://github.com/ulab-uiuc/tiny-scientiest)",
            "Accept": "application/json",
        }
        if self.s2_api_key:
            headers["X-API-KEY"] = self.s2_api_key

        try:
            print(f"[Semantic Scholar] Fetching bibtex for paper: {paper_id}")
            rsp = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                headers=headers,
                params={"fields": "citationStyles"},
                timeout=30,
            )
            print(f"[Semantic Scholar] Bibtex fetch status: {rsp.status_code}")
            rsp.raise_for_status()

            citation_styles = rsp.json().get("citationStyles", {})
            bibtex = citation_styles.get("bibtex", "N/A")

            if bibtex == "N/A":
                print(f"[WARNING] No bibtex found for paper {paper_id}")

            return bibtex

        except requests.exceptions.HTTPError as e:
            if rsp.status_code == 403:
                print(
                    "[Semantic Scholar] 403 Forbidden for bibtex fetch - API access denied"
                )
                # For 403 errors, return N/A directly without retry
                return "N/A"
            elif rsp.status_code == 429:
                print("[Semantic Scholar] 429 Rate Limited for bibtex - will retry")
                raise
            else:
                print(
                    f"[Semantic Scholar] HTTP Error {rsp.status_code} for bibtex: {e}"
                )
                raise
        except requests.exceptions.RequestException as e:
            print(f"[Semantic Scholar] Bibtex fetch failed: {e}")
            raise
        except Exception as e:
            print(f"[Semantic Scholar] Unexpected bibtex error: {e}")
            raise

    def _search_arxiv(
        self, query: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search arXiv for papers matching the query.

        Args:
            query: Search query
            result_limit: Maximum number of results

        Returns:
            List of paper dictionaries with title, abstract, authors, venue, year
        """
        try:
            import urllib.parse
            import xml.etree.ElementTree as ET

            # Clean and encode the query
            search_query = urllib.parse.quote(query)
            # Search in title and abstract
            arxiv_api_url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&max_results={result_limit}&sortBy=relevance"

            print(f"[arXiv] Searching for: {query[:60]}...")
            response = requests.get(arxiv_api_url, timeout=15)

            if response.status_code != 200:
                print(f"[arXiv] API returned status code: {response.status_code}")
                return None

            # Parse XML response
            root = ET.fromstring(response.content)

            # arXiv API uses Atom namespace
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", namespace)

            if not entries:
                print("[arXiv] No entries found")
                return None

            results = []
            for entry in entries:
                # Extract title
                title_elem = entry.find("atom:title", namespace)
                title = (
                    title_elem.text.strip().replace("\n", " ")
                    if title_elem is not None
                    else "Unknown"
                )

                # Extract abstract
                summary_elem = entry.find("atom:summary", namespace)
                abstract = summary_elem.text.strip() if summary_elem is not None else ""

                # Extract authors
                author_elems = entry.findall("atom:author/atom:name", namespace)
                authors = [
                    author.text.strip() for author in author_elems if author.text
                ]
                authors_str = " and ".join(authors) if authors else "Unknown"

                # Extract publication date
                published_elem = entry.find("atom:published", namespace)
                year = "Unknown"
                if published_elem is not None and published_elem.text:
                    # Date format: 2024-01-15T12:00:00Z
                    year = published_elem.text[:4]

                # Extract arXiv ID for potential future use
                id_elem = entry.find("atom:id", namespace)
                arxiv_id = ""
                if id_elem is not None and id_elem.text:
                    arxiv_id = id_elem.text.split("/abs/")[-1]

                paper = {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors_str,
                    "venue": "arXiv",
                    "year": year,
                    "citationCount": 0,  # arXiv doesn't provide citation count
                    "concepts": [],
                    "arxiv_id": arxiv_id,
                }
                results.append(paper)

            print(f"[arXiv] Found {len(results)} papers")
            return results

        except Exception as e:
            print(f"[arXiv] Search error: {e}")
            return None

    def _fetch_abstract_from_arxiv(self, paper_title: str) -> Optional[str]:
        """
        Fetch abstract from arXiv by searching for paper title.

        Args:
            paper_title: Title of the paper to search for

        Returns:
            Abstract text if found, None otherwise
        """
        try:
            import urllib.parse
            import xml.etree.ElementTree as ET

            # Clean and encode the title for search
            search_query = urllib.parse.quote(paper_title)
            arxiv_api_url = f"http://export.arxiv.org/api/query?search_query=ti:{search_query}&max_results=1"

            print(f"[arXiv] Searching for: {paper_title}...")
            response = requests.get(arxiv_api_url, timeout=10)

            if response.status_code != 200:
                print(f"[arXiv] API returned status code: {response.status_code}")
                return None

            # Parse XML response
            root = ET.fromstring(response.content)

            # arXiv API uses Atom namespace
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", namespace)

            if not entries:
                print("[arXiv] No entries found")
                return None

            # Get the first (best match) entry
            entry = entries[0]

            # Extract title to verify it's a good match
            entry_title = entry.find("atom:title", namespace)
            if entry_title is not None:
                entry_title_text = entry_title.text.strip().replace("\n", " ")
                print(f"[arXiv] Found: {entry_title_text[:60]}...")

            # Extract abstract
            summary = entry.find("atom:summary", namespace)
            if summary is not None:
                abstract = summary.text.strip()
                print(f"[arXiv] Abstract length: {len(abstract)} chars")
                return abstract
            else:
                print("[arXiv] No abstract found in entry")
                return None

        except Exception as e:
            print(f"[arXiv] Error fetching abstract: {e}")
            return None

    @staticmethod
    def _extract_work_info(
        work: Dict[str, Any], max_abstract_length: int = 1000
    ) -> Dict[str, str]:
        venue = next(
            (
                loc["source"]["display_name"]
                for loc in work["locations"]
                if loc["source"]
            ),
            "Unknown",
        )

        authors_list = [
            author["author"]["display_name"] for author in work["authorships"]
        ]
        authors = (
            " and ".join(authors_list)
            if len(authors_list) < 20
            else f"{authors_list[0]} et al."
        )

        abstract = work.get("abstract", "")

        # Get concepts and keywords as abstract supplement
        concepts = []
        if "concepts" in work and work["concepts"]:
            # Get top 5 most relevant concepts
            top_concepts = sorted(
                work["concepts"], key=lambda x: x.get("score", 0), reverse=True
            )[:5]
            concepts = [concept.get("display_name", "") for concept in top_concepts]

        # If abstract is too short, supplement with concepts
        if len(abstract) < 100 and concepts:
            concept_text = "Key concepts: " + ", ".join(concepts)
            abstract = abstract + ". " + concept_text if abstract else concept_text

        if len(abstract) > max_abstract_length:
            print(f"[WARNING] {work['title']}: Abstract is too long, truncating.")
            abstract = abstract[:max_abstract_length]

        return {
            "title": work["title"],
            "authors": authors,
            "venue": venue,
            "year": work.get("publication_year", "Unknown"),
            "abstract": abstract,
            "citationCount": work.get("cited_by_count", 0),
            "concepts": concepts,  # Added concept information
            "openalex_id": work.get("id", ""),  # Store OpenAlex ID for bibtex fetching
        }


app = FastMCP("tiny-scientist-paper-search", description="Paper search MCP server")


@app.tool(
    name="paper_search.run", description="Search academic papers and return metadata"
)
def run_paper_search(
    query: str,
    result_limit: int = 10,
    s2_api_key: Optional[str] = None,
    engine: Optional[str] = None,
    disable_fallback: bool = False,
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        tool = PaperSearchTool(
            s2_api_key=s2_api_key,
            engine=engine,
            disable_fallback=disable_fallback,
        )
        return tool.run(query=query, result_limit=result_limit)


if __name__ == "__main__":
    app.run()
