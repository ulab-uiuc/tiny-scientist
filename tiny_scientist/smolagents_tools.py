"""smolagents Tool implementations for tiny-scientist.

This module contains all tools migrated from MCP/FastMCP to smolagents Tool format.
"""

from __future__ import annotations

import ast
import json
import os
import os.path as osp
import re
import shutil
import tempfile
import time
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import fitz
import requests
import toml
from rich import print
from smolagents import Tool

from tiny_scientist.budget_checker import BudgetChecker
from tiny_scientist.configs import Config
from tiny_scientist.utils.error_handler import api_calling_error_exponential_backoff
from tiny_scientist.utils.llm import create_client, get_response_from_llm

# Load config
PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config.toml"
config = toml.load(CONFIG_PATH) if CONFIG_PATH.exists() else {"core": {}}


# =============================================================================
# Paper Search Tool
# =============================================================================


class PaperSearchTool(Tool):
    """Search academic papers and return metadata."""

    name = "paper_search"
    description = "Search academic papers using Semantic Scholar, OpenAlex, or arXiv and return metadata including title, abstract, authors, and bibtex."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for finding academic papers",
        },
        "result_limit": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "default": 10,
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        engine: Optional[str] = None,
        disable_fallback: bool = False,
        cost_tracker: Optional[BudgetChecker] = None,
    ) -> None:
        super().__init__()
        self.cost_tracker = cost_tracker or BudgetChecker()
        raw_key = (
            s2_api_key
            or os.environ.get("S2_API_KEY")
            or config.get("core", {}).get("s2_api_key")
        )
        self.s2_api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key
        self.disable_fallback = disable_fallback

        # Engine selection priority
        configured_engine = config.get("core", {}).get("engine")
        if engine:
            self.engine = engine
        elif configured_engine:
            self.engine = configured_engine
        elif self.s2_api_key:
            self.engine = "semanticscholar"
        else:
            self.engine = "openalex"

    def run(
        self, query: str, result_limit: Optional[int] = 10
    ) -> Dict[str, Dict[str, str]]:
        """Backward-compatible run method that calls forward."""
        return self.forward(query, result_limit)

    def forward(
        self, query: str, result_limit: Optional[int] = 10
    ) -> Dict[str, Dict[str, str]]:
        if result_limit is None:
            result_limit = 10
        results: Dict[str, Dict[str, str]] = {}
        print(f"[PaperSearchTool] Searching for: {query}")
        papers = self._search_for_papers(query, result_limit=result_limit)

        if papers:
            print(f"[PaperSearchTool] Found {len(papers)} papers")
            for i, paper in enumerate(papers):
                paper_title = paper.get("title", "Unknown Title")

                paper_data = {
                    "title": paper_title,
                    "abstract": paper.get("abstract") or "",
                    "authors": paper.get("authors") or "",
                    "venue": paper.get("venue") or "",
                    "year": paper.get("year") or "",
                    "citationCount": paper.get("citationCount", 0),
                    "concepts": paper.get("concepts", []),
                    "bibtex": "",
                }

                # Try to get bibtex from OpenAlex
                if "openalex_id" in paper and paper.get("openalex_id"):
                    bibtex = self._fetch_bibtex_from_openalex(paper["openalex_id"])
                    if bibtex:
                        paper_data["bibtex"] = bibtex

                if not paper_data["bibtex"]:
                    bibtex = self._generate_bibtex_from_metadata(paper)
                    if bibtex:
                        paper_data["bibtex"] = bibtex
                    else:
                        continue  # Skip papers without bibtex

                # Enrich with arXiv if abstract is too short
                abstract_text = paper_data["abstract"] or ""
                if len(abstract_text) < 100:
                    try:
                        arxiv_abstract = self._fetch_abstract_from_arxiv(paper_title)
                        if arxiv_abstract and len(arxiv_abstract) > len(abstract_text):
                            paper_data["abstract"] = arxiv_abstract
                    except Exception:
                        pass

                results[paper_title] = paper_data

        self.cost_tracker.report()
        return results

    def _search_for_papers(
        self, query: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        if not query:
            return None

        if self.engine == "semanticscholar":
            try:
                result = self._search_semanticscholar(query, result_limit)
                if result:
                    return result
                elif not self.disable_fallback:
                    print(
                        "[INFO] Semantic Scholar returned no results, trying arXiv..."
                    )
            except Exception as e:
                if not self.disable_fallback:
                    print(f"[WARNING] Semantic Scholar failed: {e}, trying arXiv...")
                else:
                    return None

            if not self.disable_fallback:
                try:
                    arxiv_result = self._search_arxiv(query, result_limit)
                    if arxiv_result:
                        return arxiv_result
                    return self._search_openalex(query, result_limit)
                except Exception:
                    return None
        elif self.engine == "openalex":
            try:
                result = self._search_openalex(query, result_limit)
                if result:
                    return result
                elif not self.disable_fallback:
                    print(
                        "[WARNING] OpenAlex returned no results, trying Semantic Scholar..."
                    )
            except Exception as e:
                if not self.disable_fallback:
                    print(f"[WARNING] OpenAlex failed: {e}, trying Semantic Scholar...")
                else:
                    return None

            if not self.disable_fallback:
                try:
                    return self._search_semanticscholar(query, result_limit)
                except Exception:
                    return None

        return None

    @api_calling_error_exponential_backoff(retries=3, base_wait_time=2)
    def _search_semanticscholar(
        self, query: str, result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, str | int] = {
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,paperId",
        }
        headers = {
            "User-Agent": "TinyScientist/1.0",
            "Accept": "application/json",
        }
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        rsp.raise_for_status()
        results = rsp.json()
        if not results.get("total"):
            return None
        time.sleep(2.0)
        return cast(Optional[List[Dict[str, Any]]], results.get("data"))

    def _search_openalex(
        self, query: str, result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        import pyalex
        from pyalex import Works

        mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
        if mail:
            pyalex.config.email = mail

        works = Works().search(query).get(per_page=result_limit)
        if not works:
            return None
        return [self._extract_work_info(work) for work in works]

    def _search_arxiv(
        self, query: str, result_limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        import urllib.parse
        import xml.etree.ElementTree as ET

        search_query = urllib.parse.quote(query)
        arxiv_api_url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&max_results={result_limit}&sortBy=relevance"

        response = requests.get(arxiv_api_url, timeout=15)
        if response.status_code != 200:
            return None

        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", namespace)

        if not entries:
            return None

        results = []
        for entry in entries:
            title_elem = entry.find("atom:title", namespace)
            title = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None
                else "Unknown"
            )
            summary_elem = entry.find("atom:summary", namespace)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            author_elems = entry.findall("atom:author/atom:name", namespace)
            authors = [author.text.strip() for author in author_elems if author.text]
            authors_str = " and ".join(authors) if authors else "Unknown"
            published_elem = entry.find("atom:published", namespace)
            year = "Unknown"
            if published_elem is not None and published_elem.text:
                year = published_elem.text[:4]

            paper = {
                "title": title,
                "abstract": abstract,
                "authors": authors_str,
                "venue": "arXiv",
                "year": year,
                "citationCount": 0,
                "concepts": [],
            }
            results.append(paper)

        return results

    def _fetch_bibtex_from_openalex(self, work_id: str) -> Optional[str]:
        try:
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
        except Exception:
            return None

    def _generate_bibtex_from_metadata(self, paper: Dict[str, Any]) -> str:
        try:
            title = paper.get("title", "Unknown Title")
            authors_raw = paper.get("authors", "Unknown Author")
            venue = paper.get("venue", "Unknown Venue")
            year = paper.get("year", "Unknown")

            if isinstance(authors_raw, list):
                if authors_raw and isinstance(authors_raw[0], dict):
                    author_names = [
                        a.get("name", "") for a in authors_raw if a.get("name")
                    ]
                    authors = (
                        " and ".join(author_names) if author_names else "Unknown Author"
                    )
                elif authors_raw and isinstance(authors_raw[0], str):
                    authors = " and ".join(authors_raw)
                else:
                    authors = "Unknown Author"
            elif isinstance(authors_raw, str):
                authors = authors_raw
            else:
                authors = "Unknown Author"

            clean_title = re.sub(r"[^\w\s]", "", title)
            first_word = clean_title.split()[0] if clean_title.split() else "paper"
            bibtex_key = f"{first_word.lower()}{year}"

            return f"""@article{{{bibtex_key},
    title={{{title}}},
    author={{{authors}}},
    journal={{{venue}}},
    year={{{year}}}
}}"""
        except Exception:
            return ""

    def _fetch_abstract_from_arxiv(self, paper_title: str) -> Optional[str]:
        import urllib.parse
        import xml.etree.ElementTree as ET

        search_query = urllib.parse.quote(paper_title)
        arxiv_api_url = f"http://export.arxiv.org/api/query?search_query=ti:{search_query}&max_results=1"

        response = requests.get(arxiv_api_url, timeout=10)
        if response.status_code != 200:
            return None

        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", namespace)

        if not entries:
            return None

        entry = entries[0]
        summary = entry.find("atom:summary", namespace)
        if summary is not None:
            return summary.text.strip()
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

        concepts = []
        if "concepts" in work and work["concepts"]:
            top_concepts = sorted(
                work["concepts"], key=lambda x: x.get("score", 0), reverse=True
            )[:5]
            concepts = [concept.get("display_name", "") for concept in top_concepts]

        if len(abstract) < 100 and concepts:
            concept_text = "Key concepts: " + ", ".join(concepts)
            abstract = abstract + ". " + concept_text if abstract else concept_text

        if len(abstract) > max_abstract_length:
            abstract = abstract[:max_abstract_length]

        return {
            "title": work["title"],
            "authors": authors,
            "venue": venue,
            "year": work.get("publication_year", "Unknown"),
            "abstract": abstract,
            "citationCount": work.get("cited_by_count", 0),
            "concepts": concepts,
            "openalex_id": work.get("id", ""),
        }


# =============================================================================
# Code Search Tool
# =============================================================================


class CodeSearchTool(Tool):
    """Search GitHub repositories or code snippets."""

    name = "code_search"
    description = "Search GitHub repositories or code snippets using the GitHub API."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for finding code or repositories",
        },
        "search_type": {
            "type": "string",
            "description": "Type of search: 'repositories' or 'code'",
            "default": "repositories",
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(self, cost_tracker: Optional[BudgetChecker] = None) -> None:
        super().__init__()
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.github_token = config.get("core", {}).get("github_token")

    def run(
        self, query: str, search_type: Optional[str] = "repositories"
    ) -> Dict[str, Dict[str, str]]:
        """Backward-compatible run method that calls forward."""
        return self.forward(query, search_type)

    def forward(
        self, query: str, search_type: Optional[str] = "repositories"
    ) -> Dict[str, Dict[str, str]]:
        if search_type is None:
            search_type = "repositories"
        print(f"[github API calling] Searching for code with query: {query}")
        results: Dict[str, Dict[str, str]] = {}

        try:
            idea = json.loads(query)
            if isinstance(idea, dict) and any(
                key in idea for key in ("Title", "Experiment")
            ):
                query = self._format_github_repo_query(idea)
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

    def _format_github_repo_query(
        self, idea: Dict[str, Any], max_terms: int = 6, max_query_length: int = 250
    ) -> str:
        import spacy

        title = idea.get("Title", "")
        experiment = idea.get("Experiment", "")
        combined_text = f"{title}. {experiment}"

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(combined_text)

        candidates: Set[str] = set()
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if 1 <= len(phrase.split()) <= 4:
                candidates.add(phrase)

        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 2:
                candidates.add(token.text.lower())

        seen: Set[str] = set()
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
        code_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "file_name": item["name"],
                "repository": item["repository"]["full_name"],
                "url": item["html_url"],
            }
            for item in code_results
        ]


# =============================================================================
# Drawer Tool
# =============================================================================


class DrawerTool(Tool):
    """Generate diagram SVG and summary."""

    name = "drawer"
    description = "Generate diagram SVG and summary from section content using an LLM."
    inputs = {
        "query": {
            "type": "string",
            "description": "JSON string with 'section_name' and 'section_content' fields",
        },
    }
    output_type = "object"

    def __init__(
        self,
        model: str,
        prompt_template_dir: Optional[str] = None,
        temperature: float = 0.75,
        cost_tracker: Optional[BudgetChecker] = None,
    ) -> None:
        super().__init__()
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.client, self.model = create_client(model)
        self.temperature = temperature

        # Load prompt templates
        self.config = Config(prompt_template_dir)
        self.prompts = self.config.prompt_template.drawer_prompt

        def escape_curly_braces(text: str) -> str:
            return re.sub(r"({|})", r"{{\1}}", text)

        def extract_pdf_text_from_resource(package: str, filename: str) -> str:
            with resources.files(package).joinpath(filename).open("rb") as f:
                doc = fitz.open(stream=f.read(), filetype="pdf")
                extracted = [page.get_text().strip() for page in doc]
                return "\n\n".join(extracted)

        method_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "framework.pdf"
        )
        result_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "result.pdf"
        )

        method_sample = escape_curly_braces(method_sample_raw)
        result_sample = escape_curly_braces(result_sample_raw)

        self.system_prompts = self.prompts.diagram_system_prompt.format(
            method_sample=method_sample,
            result_sample=result_sample,
        )

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        """Backward-compatible run method that calls forward."""
        return self.forward(query)

    def forward(self, query: str) -> Dict[str, Dict[str, str]]:
        try:
            query_dict = json.loads(query)
            section_name = query_dict.get("section_name")
            section_content = query_dict.get("section_content")
        except (json.JSONDecodeError, TypeError, AttributeError):
            raise ValueError(
                "Expected query to be a JSON string with 'section_name' and 'section_content'."
            )

        diagram = self._draw_diagram(
            section_name=section_name, section_content=section_content
        )

        results: Dict[str, Dict[str, str]] = {}
        if diagram:
            results["diagram"] = {
                "summary": diagram.get("summary", ""),
                "svg": diagram.get("svg", ""),
            }
        self.cost_tracker.report()
        return results

    def _draw_diagram(
        self,
        section_name: str,
        section_content: str,
        msg_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        section_prompt = self.prompts.section_prompt[section_name].format(
            section_text=section_content
        )

        llm_response, _ = get_response_from_llm(
            section_prompt,
            model=self.model,
            client=self.client,
            system_message=self.system_prompts,
            msg_history=msg_history or [],
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="generate_diagram",
        )

        return self._extract_diagram(llm_response)

    def _extract_diagram(self, response: str) -> Dict[str, Any]:
        """Extract SVG diagram from LLM response (works for all models)."""
        result: Dict[str, Any] = {"summary": "", "svg": "", "full_response": response}

        try:
            parsed = json.loads(response)
            summary = parsed["summary"]
            svg = parsed["svg"]
        except json.JSONDecodeError:
            svg_match = re.search(r"<svg.*?</svg>", response, re.DOTALL)
            svg = svg_match.group(0) if svg_match else ""
            summary = (
                re.sub(r"<svg.*?</svg>", "", response, flags=re.DOTALL)
                .strip()
                .split("\n")[0]
            )

        if "<svg" in svg and "</svg>" in svg:
            result["summary"] = summary
            result["svg"] = self._clean_svg(svg)
        return result

    def _clean_svg(self, svg: str) -> str:
        svg = svg.strip()
        svg = re.sub(r"^```(?:svg)?", "", svg)
        svg = re.sub(r"```$", "", svg)
        svg = svg.replace("&", "&amp;")
        svg = re.sub(r"<\?xml.*?\?>", "", svg, count=1)
        svg = "\n".join([line for line in svg.splitlines() if line.strip()])
        return svg.strip()


# =============================================================================
# Docker Experiment Runner (kept as class, not smolagents Tool)
# =============================================================================


class DockerExperimentRunner:
    """Docker experiment runner for executing experiments in containers."""

    def __init__(
        self,
        docker_image: str = "tiny-scientist-ml",
        docker_base: str = "python:3.11-slim",
    ):
        self.docker_image = docker_image
        self.docker_base = docker_base
        self.docker_client = None
        self.use_docker = False

        try:
            import docker

            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.use_docker = True
            print("[Docker] Docker client initialized.")
        except Exception as e:
            print(f"[Docker] Docker not available: {e}")
            self.use_docker = False

    @staticmethod
    def detect_required_packages(
        pyfile: str, base_packages: Optional[Set[str]] = None
    ) -> List[str]:
        """Detect required packages from import statements in a Python file."""
        if base_packages is None:
            base_packages = set(
                [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "matplotlib",
                    "seaborn",
                    "torch",
                    "tensorflow",
                    "transformers",
                    "datasets",
                    "evaluate",
                    "wandb",
                    "tqdm",
                    "requests",
                    "pillow",
                ]
            )

        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
        }

        stdlib_modules = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "random",
            "math",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "argparse",
            "subprocess",
            "tempfile",
            "shutil",
            "glob",
            "re",
            "typing",
            "abc",
        }

        try:
            with open(pyfile, "r") as f:
                tree = ast.parse(f.read())
        except Exception:
            return []

        pkgs: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    pkg_name = n.name.split(".")[0]
                    if pkg_name not in stdlib_modules:
                        pkgs.add(pkg_name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pkg_name = node.module.split(".")[0]
                if pkg_name not in stdlib_modules:
                    pkgs.add(pkg_name)

        mapped_pkgs = []
        for pkg in pkgs:
            if pkg not in base_packages:
                pip_pkg = package_mapping.get(pkg, pkg)
                if pip_pkg not in mapped_pkgs:
                    mapped_pkgs.append(pip_pkg)

        return sorted(mapped_pkgs)

    def get_or_build_base_image(self) -> Optional[str]:
        """Build or get the base Docker image with common ML packages."""
        if not self.use_docker or self.docker_client is None:
            return None

        from docker.errors import ImageNotFound

        try:
            self.docker_client.images.get(self.docker_image)
            print(f"[Docker] Using existing image: {self.docker_image}")
        except ImageNotFound:
            print(f"[Docker] Building image: {self.docker_image}")
            dockerfile = f"""
FROM {self.docker_base}
RUN pip install --no-cache-dir numpy pandas scikit-learn matplotlib seaborn torch tensorflow transformers datasets evaluate wandb tqdm requests pillow
"""
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                    f.write(dockerfile)
                self.docker_client.images.build(
                    path=tmpdir, tag=self.docker_image, rm=True
                )
        return self.docker_image

    def get_or_build_experiment_image(self, experiment_py_path: str) -> Optional[str]:
        """Build or get experiment-specific Docker image with required packages."""
        if not self.use_docker or self.docker_client is None:
            return None

        from docker.errors import ImageNotFound

        base_image = self.get_or_build_base_image()
        extra_pkgs = self.detect_required_packages(experiment_py_path)

        if extra_pkgs:
            image_name = f"tiny-scientist-exp-{hash(tuple(extra_pkgs))}"
            try:
                self.docker_client.images.get(image_name)
                return image_name
            except ImageNotFound:
                print(f"[Docker] Building experiment image: {image_name}")
                with tempfile.TemporaryDirectory() as tmpdir:
                    with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                        for pkg in extra_pkgs:
                            f.write(pkg + "\n")
                    dockerfile = f"""
FROM {base_image}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || exit 1
COPY experiment.py .
"""
                    with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                        f.write(dockerfile)
                    shutil.copy(
                        experiment_py_path, os.path.join(tmpdir, "experiment.py")
                    )
                    try:
                        self.docker_client.images.build(
                            path=tmpdir, tag=image_name, rm=True
                        )
                    except Exception:
                        return base_image
                return image_name
        return base_image

    def run_experiment_in_docker(
        self, experiment_code: str, run_num: int, output_dir: str, timeout: int = 7200
    ) -> Optional[Tuple[int, str]]:
        """Run experiment in a Docker container."""
        if not self.use_docker or self.docker_client is None:
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            exp_py = os.path.join(temp_dir, "experiment.py")
            with open(exp_py, "w") as f:
                f.write(experiment_code)

            image_name = self.get_or_build_experiment_image(exp_py)
            output_dir = os.path.abspath(output_dir)

            container = None
            try:
                container = self.docker_client.containers.run(
                    image=image_name,
                    command=f"python experiment.py --out_dir=run_{run_num}",
                    volumes={
                        temp_dir: {"bind": "/experiment", "mode": "rw"},
                        output_dir: {"bind": "/experiment/output", "mode": "rw"},
                    },
                    working_dir="/experiment",
                    detach=True,
                    remove=False,
                    mem_limit="2g",
                    cpu_period=100000,
                    cpu_quota=50000,
                )

                try:
                    result = container.wait(timeout=timeout)
                except Exception as wait_e:
                    try:
                        logs = container.logs().decode("utf-8")
                        container.stop(timeout=10)
                    except Exception:
                        logs = "Container failed"
                    return (1, f"Container execution failed: {wait_e}\nLogs: {logs}")

                try:
                    logs = container.logs().decode("utf-8")
                except Exception:
                    logs = "Failed to retrieve logs"

                if result["StatusCode"] == 0:
                    src = os.path.join(temp_dir, f"run_{run_num}")
                    dst = os.path.join(output_dir, f"run_{run_num}")
                    if os.path.exists(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    return (0, logs)
                else:
                    return (result["StatusCode"], logs)

            except Exception as e:
                return (1, f"Docker experiment failed: {e}")
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass

        return (1, "Docker not available")

    def cleanup_docker_images(self) -> None:
        """Clean up Docker images created during experiments."""
        if not self.use_docker or self.docker_client is None:
            return
        try:
            images = self.docker_client.images.list()
            for image in images:
                if image.tags and any("tiny-scientist" in tag for tag in image.tags):
                    self.docker_client.images.remove(image.id, force=True)
        except Exception:
            pass

    @staticmethod
    def extract_missing_package(stderr: str) -> str:
        """Extract missing package name from stderr."""
        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
        }
        for line in stderr.splitlines():
            if "ModuleNotFoundError" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    import_name = parts[1]
                    return package_mapping.get(import_name, import_name)
        return "unknown-package"


# =============================================================================
# File Tools for CodeAgent
# =============================================================================


class WriteFileTool(Tool):
    """Write content to a file in the experiment directory."""

    name = "write_file"
    description = "Write content to a file in the experiment directory. Use this to create or update experiment.py."
    inputs = {
        "filename": {
            "type": "string",
            "description": "Name of the file to write (e.g., 'experiment.py')",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
    }
    output_type = "string"

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir

    def forward(self, filename: str, content: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = osp.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {filepath}"


class ReadFileTool(Tool):
    """Read content from a file in the experiment directory."""

    name = "read_file"
    description = "Read content from a file in the experiment directory. Use this to read experiment.py or other files."
    inputs = {
        "filename": {
            "type": "string",
            "description": "Name of the file to read (e.g., 'experiment.py')",
        },
    }
    output_type = "string"

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir

    def forward(self, filename: str) -> str:
        filepath = osp.join(self.output_dir, filename)
        if not osp.exists(filepath):
            return f"File {filepath} does not exist."
        with open(filepath, "r") as f:
            content = f.read()
        return content


class RunExperimentTool(Tool):
    """Run the experiment in Docker or locally."""

    name = "run_experiment"
    description = (
        "Run the experiment.py script in Docker or locally and return the result."
    )
    inputs = {
        "run_num": {
            "type": "integer",
            "description": "The run number for this experiment execution",
        },
    }
    output_type = "string"

    def __init__(
        self, output_dir: str, docker_runner: Optional[DockerExperimentRunner] = None
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.docker_runner = docker_runner

    def forward(self, run_num: int) -> str:
        exp_path = osp.join(self.output_dir, "experiment.py")
        if not osp.exists(exp_path):
            return "Error: experiment.py does not exist. Please write it first."

        with open(exp_path, "r") as f:
            experiment_code = f.read()

        # Try Docker first
        if self.docker_runner and self.docker_runner.use_docker:
            result = self.docker_runner.run_experiment_in_docker(
                experiment_code, run_num, self.output_dir, timeout=7200
            )
            if result is not None:
                return_code, logs = result
                if return_code == 0:
                    return f"Experiment run {run_num} completed successfully.\nLogs:\n{logs}"
                else:
                    return f"Experiment run {run_num} failed with code {return_code}.\nLogs:\n{logs}"

        # Fallback to local execution
        import subprocess

        command = ["python", "experiment.py", f"--out_dir=run_{run_num}"]
        try:
            result = subprocess.run(
                command,
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if result.returncode == 0:
                return f"Experiment run {run_num} completed successfully.\nOutput:\n{result.stdout}"
            else:
                return f"Experiment run {run_num} failed.\nError:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return f"Experiment run {run_num} timed out after 7200 seconds."
        except Exception as e:
            return f"Failed to run experiment: {e}"
