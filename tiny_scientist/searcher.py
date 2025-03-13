import os
import time
from typing import Dict, List, Optional

import requests


class CodeSearcher:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

    def search_github_repositories(self, query: str, result_limit: int = 10) -> Optional[List[Dict]]:
        return self._search_github(query, result_limit, search_type="repositories")

    def search_github_code(self, query: str, result_limit: int = 10) -> Optional[List[Dict]]:
        return self._search_github(query, result_limit, search_type="code")

    def _search_github(self, query: str, result_limit: int, search_type: str) -> Optional[List[Dict]]:
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
    def _extract_github_repo_info(repos: List[Dict]) -> List[Dict]:
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
    def _extract_github_code_info(code_results: List[Dict]) -> List[Dict]:
        return [
            {
                "file_name": item["name"],
                "repository": item["repository"]["full_name"],
                "url": item["html_url"],
            }
            for item in code_results
        ]

class PaperSearcher:
    def __init__(self, s2_api_key: Optional[str] = None):
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")

    def search_for_papers(self, query: str, result_limit: int = 10, engine: str = "semanticscholar") -> Optional[List[Dict]]:
        if not query:
            return None

        if engine == "semanticscholar":
            return self._search_semanticscholar(query, result_limit)
        elif engine == "openalex":
            return self._search_openalex(query, result_limit)
        else:
            raise NotImplementedError(f"{engine=} not supported!")

    def _search_semanticscholar(self, query: str, result_limit: int) -> Optional[List[Dict]]:
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()

        results = rsp.json()
        if not results.get("total"):
            return None

        time.sleep(1.0)
        return results.get("data")

    def _search_openalex(self, query: str, result_limit: int) -> Optional[List[Dict]]:
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

    @staticmethod
    def _extract_work_info(work: any, max_abstract_length: int = 1000) -> Dict[str, str]:
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
