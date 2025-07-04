import json
import os
from typing import Any, Dict, List, Optional
import httpx
import spacy
import re
import toml
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("code_search")

# Load config
config_path = os.path.join(os.path.dirname(__file__), "../..", "config.toml")
config = toml.load(config_path) if os.path.exists(config_path) else {"core": {}}

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = config["core"].get("github_token", None)


async def make_github_request(url: str, params: dict) -> Optional[dict]:
    """Make a request to the GitHub API with proper error handling."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"GitHub API request failed: {e}")
            return None


def format_github_repo_query(idea: Dict[str, Any], max_terms: int = 6, max_query_length: int = 250) -> str:
    """Format a research idea into a GitHub search query."""
    title = idea.get("Title", "")
    experiment = idea.get("Experiment", "")
    combined_text = f"{title}. {experiment}"

    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(combined_text)
        candidates = set()

        # Extract short noun phrases
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if 1 <= len(phrase.split()) <= 4:
                candidates.add(phrase)

        # Add important standalone nouns and proper nouns
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 2:
                candidates.add(token.text.lower())

        # Clean and deduplicate
        seen = set()
        keywords = []
        for kw in candidates:
            cleaned = re.sub(r"[^\w\s]", "", kw)
            if cleaned not in seen:
                seen.add(cleaned)
                keywords.append(cleaned)
            if len(keywords) >= max_terms:
                break

        # Build query string
        quoted_keywords = [f'"{kw}"' if " " in kw else kw for kw in keywords]
        base_query = " ".join(quoted_keywords)
        suffix = " in:file language:python"
        full_query = f"{base_query} {suffix}"

        # Truncate if needed
        if len(full_query) > max_query_length:
            full_query = f"{' '.join(quoted_keywords[:max_terms//2])} {suffix}"

        return full_query
    except Exception:
        # Fallback to simple keyword extraction
        return f"{title} {experiment} language:python"


def extract_github_repo_info(repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract relevant information from GitHub repository search results."""
    return [
        {
            "name": repo["name"],
            "owner": repo["owner"]["login"],
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "url": repo["html_url"],
            "description": repo["description"] or "No description provided.",
        }
        for repo in repos
    ]


def extract_github_code_info(code_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract relevant information from GitHub code search results."""
    return [
        {
            "file_name": item["name"],
            "repository": item["repository"]["full_name"],
            "url": item["html_url"],
        }
        for item in code_results
    ]


@mcp.tool()
async def search_github_repositories(query: str, result_limit: int = 10) -> str:
    """Search GitHub repositories.

    Args:
        query: Search query string or JSON string containing research idea
        result_limit: Maximum number of results to return (default: 10)
    """
    print(f"[GitHub API] Searching repositories with query: {query}")
    
    # Try to parse as JSON (research idea format)
    try:
        idea = json.loads(query)
        if isinstance(idea, dict) and any(k in idea for k in ["Title", "Experiment"]):
            formatted_query = format_github_repo_query(idea)
            print(f"[GitHub API] Formatted query from idea: {formatted_query}")
        else:
            formatted_query = query
    except (json.JSONDecodeError, TypeError):
        formatted_query = query

    url = f"{GITHUB_API_BASE}/search/repositories"
    params = {
        "q": formatted_query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(result_limit, 100),
    }

    data = await make_github_request(url, params)
    if not data or "items" not in data:
        return json.dumps({"error": "Unable to fetch repositories or no repositories found."})

    repos = extract_github_repo_info(data["items"])
    
    # Format results for return
    results = {}
    for i, repo in enumerate(repos):
        results[str(i)] = {
            "title": repo["name"],
            "source": repo["url"],
            "info": f"Stars: {repo['stars']}, Owner: {repo['owner']}",
            "description": repo["description"]
        }

    return json.dumps(results, indent=2)


@mcp.tool()
async def search_github_code(query: str, result_limit: int = 10) -> str:
    """Search GitHub code files.

    Args:
        query: Search query string
        result_limit: Maximum number of results to return (default: 10)
    """
    print(f"[GitHub API] Searching code with query: {query}")
    
    url = f"{GITHUB_API_BASE}/search/code"
    params = {
        "q": query,
        "sort": "indexed",
        "order": "desc",
        "per_page": min(result_limit, 100),
    }

    data = await make_github_request(url, params)
    if not data or "items" not in data:
        return json.dumps({"error": "Unable to fetch code results or no code found."})

    code_results = extract_github_code_info(data["items"])
    
    # Format results for return
    results = {}
    for i, code in enumerate(code_results):
        results[str(i)] = {
            "title": code["file_name"],
            "source": code["url"],
            "info": f"Repository: {code['repository']}",
        }

    return json.dumps(results, indent=2)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 