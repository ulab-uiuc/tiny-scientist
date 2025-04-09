import abc
import os
import time
from typing import Any, Dict, List, Optional, cast

import requests
import spacy
import toml

from .utils.error_handler import api_calling_error_exponential_backoff

# Load configuration from TOML
config_path = os.path.join(os.path.dirname(__file__), "..", "config.template.toml")
config = toml.load(config_path)

nlp = spacy.load("en_core_web_sm")
# config = toml.load("config.toml")

class BaseTool(abc.ABC):

    @abc.abstractmethod
    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        pass


class CodeSearchTool(BaseTool):
    def __init__(self) -> None:
        self.github_token = config["auth"].get("github_token", None)

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        results = {}
        repos = self.search_github_repositories(query)

        if repos:
            for i, repo in enumerate(repos):
                results[str(i)] = {
                    "title": repo["name"],
                    "source": repo["url"],
                    "info": f"Stars: {repo['stars']}"
                }

        return results

    def format_github_repo_query(self, idea: Dict[str, Any], max_terms: int = 6, max_query_length: int = 250) -> str:
        import re
        title = idea.get("Title", "")
        experiment = idea.get("Experiment", "")
        combined_text = f"{title}. {experiment}"

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
        candidates = set(candidates)
        seen = set()
        keywords = []
        for kw in candidates:
            cleaned = re.sub(r"[^\w\s]", "", kw)
            if cleaned not in seen:
                seen.add(cleaned)
                keywords.append(cleaned)
            if len(keywords) >= max_terms:
                break

        # Build query string (selectively quote multi-word phrases)
        quoted_keywords = [
            f'"{kw}"' if " " in kw else kw for kw in keywords
        ]
        base_query = " ".join(quoted_keywords)
        suffix = " in:file language:python"
        full_query = f"{base_query} {suffix}"

        # Truncate if needed
        if len(full_query) > max_query_length:
            full_query = f"{' '.join(quoted_keywords[:max_terms//2])} {suffix}"

        return full_query

    def search_github_repositories(self, query: str, result_limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        return self._search_github(query, result_limit, search_type="repositories")

    def search_github_code(self, query: str, result_limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        return self._search_github(query, result_limit, search_type="code")

    def _search_github(self, query: str, result_limit: int, search_type: str) -> Optional[List[Dict[str, Any]]]:
        if search_type not in ["repositories", "code"]:
            raise ValueError("search_type must be either 'repositories' or 'code'.")

        url = f"https://api.github.com/search/{search_type}"
        headers = {"Authorization": f"token {self.github_token}"} if self.github_token else {}

        params = {
            "q": query,
            "sort": "stars" if search_type == "repositories" else "indexed",
            "order": "desc",
            "per_page": result_limit,
        }

        response = requests.get(url, headers=headers, params=params)
        print(f"GitHub {search_type.capitalize()} Response Status Code: {response.status_code}")
        response.raise_for_status()

        results = response.json()
        if "items" not in results:
            return None

        return (
            self._extract_github_repo_info(results["items"])
            if search_type == "repositories"
            else self._extract_github_code_info(results["items"])
        )

    @staticmethod
    def _extract_github_repo_info(repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    @staticmethod
    def _extract_github_code_info(code_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "file_name": item["name"],
                "repository": item["repository"]["full_name"],
                "url": item["html_url"],
            }
            for item in code_results
        ]

class PaperSearchTool(BaseTool):
    def __init__(self) -> None:
        self.s2_api_key = config["core"].get("s2_api_key", None)

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        results = {}
        papers = self.search_for_papers(query)
        if papers:
            for i, paper in enumerate(papers):

                paper_id = paper.get("paperId", None)
                bibtex = self.fetch_bibtex(paper_id) if paper_id else "N/A"

                if not bibtex or bibtex == "N/A":
                    continue

                results[paper["title"]] = {
                    "title": paper["title"],
                    # "authors": paper["authors"],
                    # "venue": paper["venue"],
                    "bibtex": bibtex
                }

        return results

    def search_for_papers(self, query: str, result_limit: int = 3) -> Optional[List[Dict[str, Any]]]:
        engine = config["thinker"].get("engine", "semanticscholar")
        if not query:
            return None

        if engine == "semanticscholar":
            return self._search_semanticscholar(query, result_limit)
        elif engine == "openalex":
            return self._search_openalex(query, result_limit)
        else:
            raise NotImplementedError(f"{engine=} not supported!")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _search_semanticscholar(self, query: str, result_limit: int) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, str | int] = {
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        }
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
            params=params,
        )
        print(f"Response Status Code: {rsp.status_code}")
        # print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()

        results = rsp.json()
        if not results.get("total"):
            return None

        time.sleep(1.0)
        return cast(Optional[List[Dict[str, Any]]], results.get("data"))  # Fix #2

    def _search_openalex(self, query: str, result_limit: int) -> Optional[List[Dict[str, Any]]]:
        import pyalex
        from pyalex import Works

        mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
        if mail:
            pyalex.config.email = mail
        else:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")

        works = Works().search(query).get(per_page=result_limit)
        if not works:
            return None

        return [self._extract_work_info(work) for work in works]

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def fetch_bibtex(self, paper_id: str) -> Any:
        rsp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
            params={"fields": "citationStyles"}
        )
        rsp.raise_for_status()
        citation_styles = rsp.json().get("citationStyles", {})
        return citation_styles.get("bibtex", "N/A")

    @staticmethod
    def _extract_work_info(work: Dict[str, Any], max_abstract_length: int = 1000) -> Dict[str, str]:
        venue = next((loc["source"]["display_name"] for loc in work["locations"] if loc["source"]), "Unknown")
        authors_list = [author["author"]["display_name"] for author in work["authorships"]]
        authors = " and ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
        abstract = work.get("abstract", "")
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
        }

    @staticmethod
    def format_paper_results(papers: Optional[List[Dict[str, Any]]]) -> str:
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                    i=i,
                    title=paper["title"],
                    authors=paper["authors"],
                    venue=paper["venue"],
                    year=paper["year"],
                    abstract=paper["abstract"],
                    cites=paper["citationCount"],
                )
            )
        return "\n\n".join(paper_strings)

    @staticmethod
    def simplify_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        simplified = []
        for paper in papers:
            raw_authors = paper.get("authors", [])
            if isinstance(raw_authors, list):
                authors_list = [
                    author["name"] if isinstance(author, dict) and "name" in author else str(author)
                    for author in raw_authors
                ]
            else:
                authors_list = [str(raw_authors)]

            if len(authors_list) > 2:
                authors_list = [authors_list[0] + " et al."]

            simplified.append({
                "year": paper.get("year"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": authors_list
            })
        return simplified
