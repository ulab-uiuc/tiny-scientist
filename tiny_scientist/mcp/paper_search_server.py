import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import toml
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("paper_search")

# Load config
config_path = os.path.join(os.path.dirname(__file__), "../..", "config.toml")
config = toml.load(config_path) if os.path.exists(config_path) else {"core": {}}

# Semantic Scholar API configuration
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = None
SEARCH_ENGINE = config["core"].get("engine", "semanticscholar")

# Debug: Print configuration status
print(f"[Paper Search] Config path: {config_path}")
print(f"[Paper Search] Config exists: {os.path.exists(config_path)}")
print(f"[Paper Search] API Key configured: {'Yes' if S2_API_KEY else 'No'}")
print(f"[Paper Search] Search engine: {SEARCH_ENGINE}")


def make_s2_request(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Make a request to the Semantic Scholar API with proper error handling."""
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}
    
    if S2_API_KEY:
        print(f"[Paper Search] Using API key: {S2_API_KEY[:10]}...")
    else:
        print("[Paper Search] Using unauthenticated access (rate limited)")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30.0)
        print(f"[Paper Search] Response status: {response.status_code}")
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        if result.get('data'):
            print(f"[Paper Search] Found {len(result['data'])} papers")
        return result
    except Exception as e:
        print(f"[Paper Search] Semantic Scholar API request failed: {e}")
        if hasattr(e, 'response'):
            print(f"[Paper Search] Response text: {e.response.text if e.response else 'No response'}")
        return None


def make_openalex_request(query: str, result_limit: int) -> Optional[List[Dict[str, Any]]]:
    """Make a request to OpenAlex API."""
    try:
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

        return [extract_openalex_work_info(work) for work in works]
    except ImportError:
        print("[ERROR] pyalex not installed, falling back to Semantic Scholar")
        return None
    except Exception as e:
        print(f"OpenAlex API request failed: {e}")
        return None


def extract_openalex_work_info(work: Dict[str, Any], max_abstract_length: int = 1000) -> Dict[str, str]:
    """Extract relevant information from OpenAlex work data."""
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
    if len(abstract) > max_abstract_length:
        print(f"[WARNING] {work['title']}: Abstract is too long, truncating.")
        abstract = abstract[:max_abstract_length]

    return {
        "title": work["title"],
        "authors": authors,
        "venue": venue,
        "year": str(work.get("publication_year", "Unknown")),
        "abstract": abstract,
        "citationCount": str(work.get("cited_by_count", 0)),
    }


@mcp.tool()
def search_papers(query: str, result_limit: int = 3) -> str:
    """Search for academic papers using Semantic Scholar or OpenAlex.

    Args:
        query: Search query string for papers
        result_limit: Maximum number of papers to return (default: 3)
    """
    print(f"[Paper Search] Searching for papers with query: {query}")
    
    if not query:
        return json.dumps({"error": "No query provided"})

    papers = None
    max_retries = 5
    retry_delay = 2
    
    # Retry logic for paper search
    for attempt in range(max_retries):
        if SEARCH_ENGINE == "semanticscholar":
            print(f"(Semantic Scholar API) Searching for papers with query: {query} (attempt {attempt + 1}/{max_retries})")
            papers = search_semanticscholar(query, result_limit)
        elif SEARCH_ENGINE == "openalex":
            print(f"(OpenAlex API) Searching for papers with query: {query} (attempt {attempt + 1}/{max_retries})")
            papers = make_openalex_request(query, result_limit)
        else:
            return json.dumps({"error": f"Unsupported search engine: {SEARCH_ENGINE}"})

        if papers and len(papers) > 0:
            print(f"✅ Papers found on attempt {attempt + 1}")
            break
        else:
            print(f"❌ No papers found on attempt {attempt + 1}")
            if attempt < max_retries - 1:  # Not the last attempt
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay += 2  # Increase delay by 2 seconds each retry

    if not papers or len(papers) == 0:
        return json.dumps({"error": f"No papers found after {max_retries} attempts"})

    # Format papers and fetch bibtex for Semantic Scholar results
    results = {}
    for i, paper in enumerate(papers):
        paper_id = paper.get("paperId", None)
        bibtex = "N/A"
        
        if SEARCH_ENGINE == "semanticscholar" and paper_id:
            bibtex = fetch_bibtex(paper_id)
            # Add delay between bibtex requests to be respectful to the API
            if i < len(papers) - 1:  # Not the last paper
                time.sleep(1.0)
        
        # Always add the paper to results, even if bibtex is not available
        title = paper.get("title", "Unknown Title")
        results[title] = {
            "title": title,
            "bibtex": bibtex
        }

    return json.dumps(results, indent=2)


def search_semanticscholar(query: str, result_limit: int) -> Optional[List[Dict[str, Any]]]:
    """Search Semantic Scholar for papers."""
    params = {
        "query": query,
        "limit": result_limit,
        "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,paperId",
    }
    
    url = f"{S2_API_BASE}/paper/search"
    data = make_s2_request(url, params)
    
    if not data or not data.get("total"):
        return None

    # Add a delay to be respectful to the API
    time.sleep(1.0)
    result = data.get("data")
    return result if isinstance(result, list) else None


@mcp.tool()
def fetch_bibtex(paper_id: str) -> str:
    """Fetch BibTeX citation for a paper by its Semantic Scholar ID.

    Args:
        paper_id: Semantic Scholar paper ID
    """
    print(f"[Paper Search] Fetching BibTeX for paper ID: {paper_id}")
    
    url = f"{S2_API_BASE}/paper/{paper_id}"
    params = {"fields": "citationStyles"}
    
    data = make_s2_request(url, params)
    if not data:
        return "N/A"
    
    citation_styles = data.get("citationStyles", {})
    bibtex = citation_styles.get("bibtex", "N/A")
    return bibtex if isinstance(bibtex, str) else "N/A"


@mcp.tool()
def get_paper_details(paper_id: str) -> str:
    """Get detailed information about a paper by its Semantic Scholar ID.

    Args:
        paper_id: Semantic Scholar paper ID
    """
    print(f"[Paper Search] Getting details for paper ID: {paper_id}")
    
    url = f"{S2_API_BASE}/paper/{paper_id}"
    params = {"fields": "title,authors,venue,year,abstract,citationCount,citationStyles"}
    
    data = make_s2_request(url, params)
    if not data:
        return json.dumps({"error": "Paper not found or API error"})
    
    return json.dumps(data, indent=2)


# Import asyncio at the end to avoid issues

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 