from __future__ import annotations

import abc
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import toml
from fastmcp import FastMCP  # type: ignore
from rich import print

import tiny_scientist
from tiny_scientist.budget_checker import BudgetChecker

PACKAGE_ROOT = Path(tiny_scientist.__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config.toml"
config = toml.load(CONFIG_PATH) if CONFIG_PATH.exists() else {"core": {}}


class BaseTool(abc.ABC):
    def __init__(self, cost_tracker: Optional[BudgetChecker] = None) -> None:
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.github_token = config["core"].get("github_token")

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, str]]:
        raise NotImplementedError


class CodeSearchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__()

    def run(
        self, query: str, search_type: str = "repositories"
    ) -> Dict[str, Dict[str, str]]:
        print(f"[github API calling] Searching for code with query: {query}")
        results: Dict[str, Dict[str, str]] = {}

        try:
            idea = json.loads(query)
            if isinstance(idea, dict) and any(
                key in idea for key in ("Title", "Experiment")
            ):
                query = self.format_github_repo_query(idea)
                print(f"[github API calling] Formatted query from idea: {query}")
        except (json.JSONDecodeError, TypeError):
            pass

        repos = self._search_github(query=query, search_type=search_type)
        if repos:
            for i, repo in enumerate(repos):
                results[str(i)] = {
                    "title": repo["name"],
                    "source": repo["url"],
                    "info": f"Stars: {repo['stars']}",
                }

        self.cost_tracker.report()
        return results

    def format_github_repo_query(
        self, idea: Dict[str, Any], max_terms: int = 6, max_query_length: int = 250
    ) -> str:
        import spacy

        title = idea.get("Title", "")
        experiment = idea.get("Experiment", "")
        combined_text = f"{title}. {experiment}"

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(combined_text)

        candidates = set()
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if 1 <= len(phrase.split()) <= 4:
                candidates.add(phrase)

        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 2:
                candidates.add(token.text.lower())

        seen = set()
        keywords: List[str] = []
        for kw in candidates:
            cleaned = re.sub(r"[^\w\s]", "", kw)
            if cleaned not in seen:
                seen.add(cleaned)
                keywords.append(cleaned)
            if len(keywords) >= max_terms:
                break

        quoted_keywords = [f'"{kw}"' if " " in kw else kw for kw in keywords]
        base_query = " ".join(quoted_keywords)
        suffix = " in:file language:python"
        full_query = f"{base_query} {suffix}"

        if len(full_query) > max_query_length:
            full_query = f"{' '.join(quoted_keywords[: max_terms // 2])} {suffix}"

        return full_query

    def _search_github(
        self, query: str, search_type: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        if search_type not in ("repositories", "code"):
            raise ValueError("search_type must be either 'repositories' or 'code'.")

        url = f"https://api.github.com/search/{search_type}"
        headers = (
            {"Authorization": f"token {self.github_token}"} if self.github_token else {}
        )
        params = {
            "q": query,
            "sort": "stars" if search_type == "repositories" else "indexed",
            "order": "desc",
            "per_page": result_limit,
        }

        response = requests.get(url, headers=headers, params=params, timeout=60)
        print(
            f"GitHub {search_type.capitalize()} Response Status Code: {response.status_code}"
        )
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
                "description": repo.get("description") or "No description provided.",
            }
            for repo in repos
        ]

    @staticmethod
    def _extract_github_code_info(
        code_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "file_name": item["name"],
                "repository": item["repository"]["full_name"],
                "url": item["html_url"],
            }
            for item in code_results
        ]


app = FastMCP("tiny-scientist-code-search", description="Code search MCP server")


@app.tool(
    name="code_search.run", description="Search GitHub repositories or code snippets"
)
def run_code_search(
    query: str, search_type: str = "repositories"
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        tool = CodeSearchTool()
        return tool.run(query=query, search_type=search_type)


if __name__ == "__main__":
    app.run()
