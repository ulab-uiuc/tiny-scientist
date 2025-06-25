"""
FastMCP Server for Tiny Scientist Tools

This module provides a Model Context Protocol (MCP) server that exposes
the research tools from the tiny-scientist project as MCP tools, resources,
and prompts. It includes paper search, code search, and diagram generation
capabilities.

Based on the original tool.py implementation.
"""

import json
import os
import re
import time
import asyncio
from typing import Any, Dict, List, Optional, cast

import requests
import toml
from fastmcp import FastMCP, Context

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "../tiny_scientist/config.toml")
config = toml.load(config_path) if os.path.exists(config_path) else {"core": {}}

# Create FastMCP server instance
mcp = FastMCP("TinyScientistServer")

# Retry decorator for API calls
def api_calling_error_exponential_backoff(retries=5, base_wait_time=2):
    """Decorator for exponential backoff retry logic (async version)"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    wait_time = base_wait_time * (2 ** attempt)
                    await asyncio.sleep(wait_time)  # Use async sleep
            return None
        return wrapper
    return decorator

# ============================================================================
# Paper Search Tools
# ============================================================================

@mcp.tool()
async def search_papers(
    query: str, 
    ctx: Context,
    result_limit: int = 3
) -> Dict[str, Any]:
    """
    Search for academic papers using Semantic Scholar or OpenAlex
    
    Args:
        query: Search query string for academic papers
        result_limit: Maximum number of papers to return (default: 3)
        ctx: MCP context for logging and progress reporting
        
    Returns:
        Dictionary containing search results with paper information
    """
    await ctx.info(f"Starting paper search for query: {query}")
    
    if not query:
        await ctx.error("Query string cannot be empty")
        return {"error": "Query string cannot be empty"}

    s2_api_key = config["core"].get("s2_api_key", None)
    engine = config["core"].get("engine", "semanticscholar")
    
    results = {}
    papers = None
    
    try:
        await ctx.info(f"Using search engine: {engine}")
        
        if engine == "semanticscholar":
            papers = await _search_semanticscholar_async(query, result_limit, s2_api_key, ctx)
        elif engine == "openalex":
            papers = await _search_openalex_async(query, result_limit, ctx)
        else:
            error_msg = f"Unsupported search engine: {engine}"
            await ctx.error(error_msg)
            return {"error": error_msg}

        if papers:
            await ctx.info(f"Found {len(papers)} papers, processing...")
            
            for i, paper in enumerate(papers):
                await ctx.report_progress(i, len(papers), f"Processing paper {i+1}")
                
                paper_id = paper.get("paperId", None)
                bibtex = await _fetch_bibtex_async(paper_id, s2_api_key, ctx) if paper_id else "N/A"

                if not bibtex or bibtex == "N/A":
                    continue

                results[paper["title"]] = {
                    "title": paper["title"], 
                    "bibtex": bibtex,
                    "authors": paper.get("authors", "Unknown"),
                    "year": paper.get("year", "Unknown"),
                    "venue": paper.get("venue", "Unknown"),
                    "citationCount": paper.get("citationCount", 0),
                    "paperId": paper.get("paperId", ""),
                    "abstract": paper.get("abstract", "")[:500] + "..."  # Truncate abstract
                }

        await ctx.info(f"Paper search completed successfully with {len(results)} results")
        return {"results": results, "total_found": len(results), "query": query}
        
    except Exception as e:
        error_msg = f"Error during paper search: {str(e)}"
        await ctx.error(error_msg)
        return {"error": error_msg}

async def _search_semanticscholar_async(
    query: str, 
    result_limit: int, 
    s2_api_key: Optional[str],
    ctx: Context
) -> Optional[List[Dict[str, Any]]]:
    """Search papers using Semantic Scholar API"""
    
    await ctx.info("Calling Semantic Scholar API...")
    
    params: Dict[str, str | int] = {
        "query": query,
        "limit": result_limit,
        "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
    }

    headers = {"X-API-KEY": s2_api_key} if s2_api_key else {}
    
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
            timeout=30  # Add timeout
        )
        
        await ctx.info(f"Semantic Scholar API response status: {response.status_code}")
        response.raise_for_status()

        results = response.json()
        if not results.get("total"):
            await ctx.info("No papers found in Semantic Scholar")
            return None

        await asyncio.sleep(1.0)  # Rate limiting - use async sleep
        return cast(Optional[List[Dict[str, Any]]], results.get("data"))
        
    except requests.exceptions.RequestException as e:
        await ctx.error(f"Semantic Scholar API request failed: {str(e)}")
        return None
    except Exception as e:
        await ctx.error(f"Unexpected error in Semantic Scholar search: {str(e)}")
        return None

async def _search_openalex_async(
    query: str, 
    result_limit: int,
    ctx: Context
) -> Optional[List[Dict[str, Any]]]:
    """Search papers using OpenAlex API"""
    
    await ctx.info("Calling OpenAlex API...")
    
    try:
        import pyalex
        from pyalex import Works

        mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
        if mail:
            pyalex.config.email = mail
        else:
            await ctx.info("WARNING: Please set OPENALEX_MAIL_ADDRESS for better API access")

        works = Works().search(query).get(per_page=result_limit)
        if not works:
            await ctx.info("No papers found in OpenAlex")
            return None

        return [_extract_work_info(work) for work in works]
    except ImportError:
        await ctx.error("pyalex library not installed")
        return None

async def _fetch_bibtex_async(
    paper_id: str, 
    s2_api_key: Optional[str],
    ctx: Context
) -> str:
    """Fetch BibTeX citation for a paper"""
    
    headers = {"X-API-KEY": s2_api_key} if s2_api_key else {}
    
    try:
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            headers=headers,
            params={"fields": "citationStyles"},
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        citation_styles = response.json().get("citationStyles", {})
        bibtex = citation_styles.get("bibtex", "N/A")
        
        if bibtex != "N/A":
            await ctx.info(f"Successfully fetched BibTeX for paper {paper_id}")
        
        return bibtex
    except requests.exceptions.RequestException as e:
        await ctx.error(f"Failed to fetch BibTeX for paper {paper_id}: {str(e)}")
        return "N/A"
    except Exception as e:
        await ctx.error(f"Unexpected error fetching BibTeX for paper {paper_id}: {str(e)}")
        return "N/A"

def _extract_work_info(work: Dict[str, Any], max_abstract_length: int = 1000) -> Dict[str, str]:
    """Extract work information from OpenAlex response"""
    
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
        abstract = abstract[:max_abstract_length]

    return {
        "title": work["title"],
        "authors": authors,
        "venue": venue,
        "year": work.get("publication_year", "Unknown"),
        "abstract": abstract,
        "citationCount": work.get("cited_by_count", 0),
        "paperId": work.get("id", "")
    }

@mcp.tool()
async def get_paper_bibtex(paper_id: str, ctx: Context) -> Dict[str, Any]:
    """
    Get BibTeX citation for a specific paper
    
    Args:
        paper_id: The paper ID (from Semantic Scholar)
        ctx: MCP context for logging
        
    Returns:
        Dictionary containing BibTeX citation information
    """
    await ctx.info(f"Fetching BibTeX for paper ID: {paper_id}")
    
    try:
        s2_api_key = config["core"].get("s2_api_key", None)
        bibtex = await _fetch_bibtex_async(paper_id, s2_api_key, ctx)
        
        if bibtex and bibtex != "N/A":
            await ctx.info("Successfully retrieved BibTeX citation")
            return {"bibtex": bibtex, "paper_id": paper_id}
        else:
            error_msg = f"Unable to retrieve BibTeX for paper {paper_id}"
            await ctx.error(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error retrieving BibTeX: {str(e)}"
        await ctx.error(error_msg)
        return {"error": error_msg}

# ============================================================================
# GitHub Code Search Tools  
# ============================================================================

@mcp.tool()
async def search_github_repositories(
    query: str, 
    ctx: Context,
    result_limit: int = 10
) -> Dict[str, Any]:
    """
    Search for GitHub repositories
    
    Args:
        query: Search query for GitHub repositories
        ctx: MCP context for logging
        result_limit: Maximum number of repositories to return (default: 10)
        
    Returns:
        Dictionary containing repository search results
    """
    await ctx.info(f"Searching GitHub repositories for: {query}")
    
    return await _search_github_async(query, "repositories", result_limit, ctx)

@mcp.tool()
async def search_github_code(
    query: str, 
    ctx: Context,
    result_limit: int = 10
) -> Dict[str, Any]:
    """
    Search for code in GitHub repositories
    
    Args:
        query: Search query for GitHub code
        ctx: MCP context for logging
        result_limit: Maximum number of code results to return (default: 10)
        
    Returns:
        Dictionary containing code search results
    """
    await ctx.info(f"Searching GitHub code for: {query}")
    
    return await _search_github_async(query, "code", result_limit, ctx)

async def _search_github_async(
    query: str, 
    search_type: str, 
    result_limit: int,
    ctx: Context
) -> Dict[str, Any]:
    """Generic GitHub search function"""
    
    if search_type not in ["repositories", "code"]:
        error_msg = "search_type must be either 'repositories' or 'code'"
        await ctx.error(error_msg)
        return {"error": error_msg}

    github_token = config["core"].get("github_token", None)
    url = f"https://api.github.com/search/{search_type}"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}

    params = {
        "q": query,
        "sort": "stars" if search_type == "repositories" else "indexed",
        "order": "desc",
        "per_page": result_limit,
    }

    try:
        await ctx.info(f"Calling GitHub {search_type} API...")
        
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=30  # Add timeout
        )
        await ctx.info(f"GitHub API Response Status: {response.status_code}")
        
        if response.status_code == 403:
            await ctx.error("GitHub API rate limit exceeded. Consider adding a GitHub token.")
            return {"error": "GitHub API rate limit exceeded"}
        
        response.raise_for_status()

        results = response.json()
        if "items" not in results:
            await ctx.info("No results found")
            return {"results": {}, "total_found": 0}

        processed_results = (
            _extract_github_repo_info(results["items"])
            if search_type == "repositories"
            else _extract_github_code_info(results["items"])
        )

        # Convert to numbered dictionary format
        formatted_results = {}
        for i, item in enumerate(processed_results):
            if search_type == "repositories":
                formatted_results[str(i)] = {
                    "title": item["name"],
                    "source": item["url"],
                    "info": f"Stars: {item['stars']}, Forks: {item['forks']}",
                    "description": item["description"]
                }
            else:
                formatted_results[str(i)] = {
                    "title": item["file_name"],
                    "source": item["url"],
                    "info": f"Repository: {item['repository']}"
                }

        await ctx.info(f"Found {len(formatted_results)} {search_type} results")
        return {"results": formatted_results, "total_found": len(formatted_results)}

    except requests.exceptions.RequestException as e:
        error_msg = f"GitHub API request failed: {str(e)}"
        await ctx.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error searching GitHub {search_type}: {str(e)}"
        await ctx.error(error_msg)
        return {"error": error_msg}

def _extract_github_repo_info(repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract repository information from GitHub API response"""
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

def _extract_github_code_info(code_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract code information from GitHub API response"""
    return [
        {
            "file_name": item["name"],
            "repository": item["repository"]["full_name"],
            "url": item["html_url"],
        }
        for item in code_results
    ]

@mcp.tool()
async def format_github_query_from_idea(
    idea_json: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Format a GitHub search query from research idea JSON
    
    Args:
        idea_json: JSON string containing research idea with Title and Experiment fields
        ctx: MCP context for logging
        
    Returns:
        Formatted GitHub search query
    """
    await ctx.info("Formatting GitHub query from research idea")
    
    try:
        idea = json.loads(idea_json)
        if not isinstance(idea, dict) or not any(k in idea for k in ["Title", "Experiment"]):
            error_msg = "Invalid idea format. Expected JSON with 'Title' and/or 'Experiment' fields"
            await ctx.error(error_msg)
            return {"error": error_msg}
        
        formatted_query = _format_github_repo_query(idea)
        await ctx.info(f"Generated formatted query: {formatted_query}")
        
        return {"formatted_query": formatted_query, "original_idea": idea}
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {str(e)}"
        await ctx.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error formatting query: {str(e)}"
        await ctx.error(error_msg)
        return {"error": error_msg}

def _format_github_repo_query(
    idea: Dict[str, Any], 
    max_terms: int = 6, 
    max_query_length: int = 250
) -> str:
    """Format GitHub repository query from idea dictionary"""
    
    try:
        import spacy
        
        title = idea.get("Title", "")
        experiment = idea.get("Experiment", "")
        combined_text = f"{title}. {experiment}"

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
        
    except ImportError:
        # Fallback without spacy
        title = idea.get("Title", "")
        experiment = idea.get("Experiment", "")
        combined_text = f"{title} {experiment}"
        
        # Simple keyword extraction
        words = re.findall(r'\b\w{3,}\b', combined_text.lower())
        keywords = list(set(words))[:max_terms]
        
        return " ".join(keywords) + " in:file language:python"

# ============================================================================
# Resources
# ============================================================================

@mcp.resource("research://papers/recent")
def get_recent_papers():
    """Get information about recently searched papers"""
    return "Recent paper search history and cached results would be available here"

@mcp.resource("research://config/settings")
def get_config_settings():
    """Get current configuration settings"""
    return json.dumps({
        "search_engines": {
            "current": config["core"].get("engine", "semanticscholar"),
            "available": ["semanticscholar", "openalex"]
        },
        "api_keys_configured": {
            "semantic_scholar": bool(config["core"].get("s2_api_key")),
            "github": bool(config["core"].get("github_token"))
        }
    }, indent=2)

@mcp.resource("research://github/trending/{language}")
def get_trending_repositories(language: str):
    """Get trending repositories for a specific programming language"""
    return f"Trending {language} repositories would be listed here"

# ============================================================================
# Prompts
# ============================================================================

@mcp.prompt()
def research_paper_search_prompt(topic: str, focus_area: str = "general") -> str:
    """
    Generate a prompt for academic paper search
    
    Args:
        topic: Research topic
        focus_area: Specific focus area (default: general)
        
    Returns:
        Formatted search prompt
    """
    return f"""Please search for recent academic papers on the topic of '{topic}' with a focus on '{focus_area}'. 
    
I'm particularly interested in:
- High-impact publications from the last 3 years
- Papers with significant citation counts
- Novel methodologies and breakthrough findings
- Comprehensive review papers that provide good overviews

Please prioritize papers from reputable conferences and journals in the field."""

@mcp.prompt()
def code_implementation_search_prompt(algorithm: str, language: str = "python") -> str:
    """
    Generate a prompt for searching code implementations
    
    Args:
        algorithm: Algorithm or technique to search for
        language: Programming language (default: python)
        
    Returns:
        Formatted code search prompt
    """
    return f"""Help me find high-quality implementations of '{algorithm}' in {language}. 

I'm looking for:
- Well-documented repositories with clear README files
- Implementations with good test coverage
- Code that follows best practices and is actively maintained
- Examples and tutorials that demonstrate usage
- Repositories with significant community engagement (stars, forks, issues)

Please focus on repositories that would be suitable for learning and adaptation in research projects."""

@mcp.prompt()
def research_idea_development_prompt(domain: str, current_gaps: str = "") -> str:
    """
    Generate a prompt for research idea development
    
    Args:
        domain: Research domain
        current_gaps: Known gaps in current research (optional)
        
    Returns:
        Formatted research development prompt
    """
    gaps_section = f"\n\nKnown research gaps:\n{current_gaps}" if current_gaps else ""
    
    return f"""I'm working on developing research ideas in the domain of '{domain}'. 

Please help me:
1. Identify current trends and hot topics in this field
2. Find recent breakthrough papers that might inspire new directions
3. Discover potential research gaps that haven't been fully explored
4. Locate relevant code repositories and datasets that could support new research
5. Understand the key methodologies currently being used{gaps_section}

The goal is to develop novel, impactful research ideas that build upon existing work while addressing important open problems."""

# ============================================================================
# Server Configuration and Startup
# ============================================================================

if __name__ == "__main__":
    # Run the MCP server
    print("Starting Tiny Scientist MCP Server...")
    print("Available tools: search_papers, get_paper_bibtex, search_github_repositories, search_github_code")
    print("Available resources: research://papers/recent, research://config/settings, research://github/trending/{language}")
    print("Available prompts: research_paper_search_prompt, code_implementation_search_prompt, research_idea_development_prompt")
    
    mcp.run()