import os
import time
from typing import Dict, List, Optional, Any
import re
import requests

from tiny_scientist.utils.error_handler import api_calling_error_exponential_backoff


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


class CitationSearch:
    def __init__(self, 
                 s2_api_key: Optional[str] = None):
        self.s2_api_key = s2_api_key

    def _search_semanticscholar(self, query: str, result_limit: int = 1) -> Optional[List[Dict[str, Any]]]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {"X-API-KEY": self.s2_api_key} if self.s2_api_key else {}
        params = {
            "query": query,
            "limit": result_limit,
            "fields": "paperId,title,authors,venue,year,abstract,citationStyles,citationCount",
        }
        try:
            rsp = requests.get(url, headers=headers, params=params)
            rsp.raise_for_status()
            results = rsp.json()
            total = results.get("total", 0)
            time.sleep(1.0)  # To respect rate limits
            if total > 0:
                return results.get("data")
            else:
                return None
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return None

    def _fetch_bibtex(self, paper_id: str) -> Optional[str]:
        """
        Fetch the BibTeX entry for a given paper ID from Semantic Scholar.
        """
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        headers = {"X-API-KEY": self.s2_api_key} if self.s2_api_key else {}
        params = {"fields": "citationStyles"}
        try:
            rsp = requests.get(base_url, headers=headers, params=params)
            rsp.raise_for_status()
            results = rsp.json()
            if "citationStyles" in results and "bibtex" in results["citationStyles"]:
                return results["citationStyles"]["bibtex"]
            else:
                print(f"No BibTeX found for paper ID: {paper_id}")
                return None
        except Exception as e:
            print(f"Error fetching BibTeX for paper ID {paper_id}: {e}")
            return None

    def get_bibtex_by_query(self, query: str) -> Optional[str]:
        results = self._search_semanticscholar(query, result_limit=1)
        if not results:
            print("No papers found for query:", query)
            return None

        top_paper = results[0]
        paper_id = top_paper.get("paperId")
        if not paper_id:
            print("Paper ID not found for the top result.")
            return None

        bibtex = self._fetch_bibtex(paper_id)
        return bibtex
    
    def extract_citation_ids(self, text: str) -> List[str]:
        patterns = [
            r"\\cite\{([^}]+)\}",
            r"\\citep\{([^}]+)\}",
            r"\\citet\{([^}]+)\}"
        ]
        citation_ids = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, text)

            for match in matches:
                keys = [key.strip() for key in match.split(",")]
                citation_ids.update(keys)
        
        return list(citation_ids)
    
    def get_bibtex_for_ids(self, text: str) -> List[str]:
        citation_ids = self.extract_citation_ids(text)
        bibtex_entries = []

        for cid in citation_ids:
            bibtex = self.get_bibtex_by_query(cid)
            if bibtex:
                bibtex_entries.append(bibtex)
            else:
                print(f"BibTeX entry not found for citation ID: {cid}")
        
        return bibtex_entries
    

class PaperSearcher:
    def __init__(self, s2_api_key: Optional[str] = None):
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")

    def search_for_papers(self, query: str, result_limit: int = 1, engine: str = "semanticscholar") -> Optional[List[Dict]]:
        if not query:
            return None

        if engine == "semanticscholar":
            return self._search_semanticscholar(query, result_limit)
        elif engine == "openalex":
            return self._search_openalex(query, result_limit)
        else:
            raise NotImplementedError(f"{engine=} not supported!")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
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
    
    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _fetch_bibtex(self, paper_id: str) -> Optional[str]:
        """Fetches BibTeX entry from Semantic Scholar using paper ID."""
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

        rsp = requests.get(
            base_url,
            headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
            params={"fields": "citationStyles"},
        )

        print(f"BibTeX Fetch Response Status: {rsp.status_code}")
        print(f"BibTeX Fetch Response Content: {rsp.text[:500]}")

        if rsp.status_code != 200:
            print(f"Failed to fetch BibTeX for paper ID: {paper_id}")
            return None

        results = rsp.json()
        if "citationStyles" in results and "bibtex" in results["citationStyles"]:
            return results["citationStyles"]["bibtex"]

        print(f"No BibTeX found for paper ID: {paper_id}")
        return None

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

    @staticmethod
    def format_paper_results(papers: Optional[List[Dict]]) -> str:
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
                    cites=paper["citationCount"],
                    abstract=paper["abstract"],
                )
            )
        
        return "\n\n".join(paper_strings)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _fetch_bibtex(self, paper_id: str) -> Optional[str]:
        """Fetches BibTeX entry from Semantic Scholar using paper ID."""
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

        rsp = requests.get(
            base_url,
            headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
            params={"fields": "citationStyles"},
        )

        print(f"BibTeX Fetch Response Status: {rsp.status_code}")
        print(f"BibTeX Fetch Response Content: {rsp.text[:500]}")

        if rsp.status_code != 200:
            print(f"Failed to fetch BibTeX for paper ID: {paper_id}")
            return None

        results = rsp.json()
        if "citationStyles" in results and "bibtex" in results["citationStyles"]:
            return results["citationStyles"]["bibtex"]

        print(f"No BibTeX found for paper ID: {paper_id}")
        return None

