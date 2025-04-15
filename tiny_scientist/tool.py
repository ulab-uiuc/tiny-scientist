import abc
import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional, cast

import requests
import toml
import yaml
from rich import print

from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import create_client, get_response_from_llm

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
config = toml.load(config_path) if os.path.exists(config_path) else {"core": {}}


class BaseTool(abc.ABC):
    @abc.abstractmethod
    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        pass


class CodeSearchTool(BaseTool):
    def __init__(self) -> None:
        self.github_token = config["core"].get("github_token", None)

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        print(f"Searching for code with query: {query}")
        results = {}
        repos = self.search_github_repositories(query)

        if repos:
            for i, repo in enumerate(repos):
                results[str(i)] = {
                    "title": repo["name"],
                    "source": repo["url"],
                    "info": f"Stars: {repo['stars']}",
                }

        return results

    def format_github_repo_query(
        self, idea: Dict[str, Any], max_terms: int = 6, max_query_length: int = 250
    ) -> str:
        import re

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

    def search_github_repositories(
        self, query: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        return self._search_github(query, result_limit, search_type="repositories")

    def search_github_code(
        self, query: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        return self._search_github(query, result_limit, search_type="code")

    def _search_github(
        self, query: str, result_limit: int, search_type: str
    ) -> Optional[List[Dict[str, Any]]]:
        if search_type not in ["repositories", "code"]:
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

        response = requests.get(url, headers=headers, params=params)
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
                "description": repo["description"] or "No description provided.",
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

                results[paper["title"]] = {"title": paper["title"], "bibtex": bibtex}

        return results

    def search_for_papers(
        self, query: str, result_limit: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        print(f"Searching for papers with query: {query}")
        if not query:
            return None

        engine = config["core"].get("engine", "semanticscholar")
        if engine == "semanticscholar":
            return self._search_semanticscholar(query, result_limit)
        elif engine == "openalex":
            return self._search_openalex(query, result_limit)
        else:
            raise NotImplementedError(f"{engine=} not supported!")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _search_semanticscholar(
        self, query: str, result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, str | int] = {
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        }

        headers = {"X-API-KEY": self.s2_api_key} if self.s2_api_key else {}
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
        )
        rsp.raise_for_status()

        results = rsp.json()
        if not results.get("total"):
            return None

        time.sleep(1.0)
        return cast(Optional[List[Dict[str, Any]]], results.get("data"))

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

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def fetch_bibtex(self, paper_id: str) -> Any:
        headers = {"X-API-KEY": self.s2_api_key} if self.s2_api_key else {}
        rsp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            headers=headers,
            params={"fields": "citationStyles"},
        )
        rsp.raise_for_status()
        citation_styles = rsp.json().get("citationStyles", {})
        return citation_styles.get("bibtex", "N/A")

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
                f"""{i}: {paper["title"]}. {paper["authors"]}. {paper["venue"]}, {paper["year"]}.
Number of citations: {paper["citationCount"]}
Abstract: {paper["abstract"]}"""
            )

        return "\n\n".join(paper_strings)

    @staticmethod
    def simplify_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        simplified = []
        for paper in papers:
            raw_authors = paper.get("authors", [])

            if isinstance(raw_authors, list):
                authors_list = [
                    author["name"]
                    if isinstance(author, dict) and "name" in author
                    else str(author)
                    for author in raw_authors
                ]
            else:
                authors_list = [str(raw_authors)]

            if len(authors_list) > 2:
                authors_list = [authors_list[0] + " et al."]

            simplified.append(
                {
                    "year": paper.get("year"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "authors": authors_list,
                }
            )

        return simplified


class DrawerTool(BaseTool):
    def __init__(self, model: Any, prompt_template_dir: str, temperature: float = 0.75):
        self.client, self.model = create_client(model)
        self.temperature = temperature

        # Load prompt templates
        with open(osp.join(prompt_template_dir, "diagram_prompt.yaml"), "r") as f:
            self.prompts = yaml.safe_load(f)

        # Process template instructions
        if (
            "template_instructions" in self.prompts
            and "few_shot_instructions" in self.prompts
        ):
            self.prompts["few_shot_instructions"] = self.prompts[
                "few_shot_instructions"
            ].replace(
                "{{ template_instructions }}", self.prompts["template_instructions"]
            )

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        diagram = self.draw_diagram(query)
        results = {}
        if diagram:
            results["diagram"] = {
                "summary": diagram.get("summary", ""),
                "svg": diagram.get("svg", ""),
            }
        return results

    def draw_diagram(
        self,
        text: str,
        example: Optional[str] = None,
        msg_history: Optional[List[Dict[str, Any]]] = None,
        return_msg_history: bool = False,
        drawer_system_prompt: Optional[str] = None,
    ) -> Any:
        # Use default system prompt if none provided
        drawer_system_prompt = drawer_system_prompt or self.prompts.get(
            "diagram_system_prompt_base"
        )

        # Prepare prompt with the few-shot example
        base_prompt = self._prepare_diagram_prompt(text, example)

        # Generate diagram
        diagram, updated_msg_history = self._generate_diagram(
            base_prompt, drawer_system_prompt, msg_history
        )

        return (diagram, updated_msg_history) if return_msg_history else diagram

    def _prepare_diagram_prompt(self, text: str, example: Optional[str] = None) -> str:
        if example:
            # Format with the example
            few_shot_prompt = self.prompts["few_shot_instructions"].format(
                example=example
            )
            base_prompt = f"{few_shot_prompt}\n\nHere is the paper you are asked to create a diagram for:\n```\n{text}\n```"
        else:
            # Use just the template instructions
            base_prompt = f"{self.prompts['template_instructions']}\n\nHere is the paper you are asked to create a diagram for:\n```\n{text}\n```"

        return str(base_prompt)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_diagram(
        self,
        base_prompt: str,
        drawer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Ensure msg_history is a list
        msg_history = msg_history or []

        # Generate diagram
        llm_response, msg_history = get_response_from_llm(
            base_prompt,
            model=self.model,
            client=self.client,
            system_message=drawer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=self.temperature,
        )

        # Extract the diagram from the response
        diagram = self._extract_diagram(llm_response)

        return diagram, msg_history

    def _extract_diagram(self, response: str) -> Dict[str, Any]:
        result = {"summary": "", "svg": "", "full_response": response}

        # Extract the summary
        summary_start = response.find("SUMMARY:")
        if summary_start != -1:
            summary_end = response.find("DIAGRAM SVG:", summary_start)
            if summary_end != -1:
                result["summary"] = response[summary_start + 8 : summary_end].strip()

        # Extract the SVG
        svg_start = response.find("```svg", summary_start if summary_start != -1 else 0)
        if svg_start == -1:
            # Try without language specifier
            svg_start = response.find(
                "```", summary_start if summary_start != -1 else 0
            )
            if svg_start != -1:
                svg_start += 3  # Skip past ```
        else:
            svg_start += 6  # Skip past ```svg

        if svg_start != -1:
            svg_end = response.find("```", svg_start)
            if svg_end != -1:
                result["svg"] = response[svg_start:svg_end].strip()

        return result
