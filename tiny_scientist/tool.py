import abc
import ast
import json
import os
import re
import shutil
import tempfile
import time
from importlib import resources
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import docker
import fitz
import requests
import toml
from docker.errors import ImageNotFound
from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import create_client, get_response_from_llm

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
config = toml.load(config_path) if os.path.exists(config_path) else {"core": {}}


class BaseTool(abc.ABC):
    def __init__(self, cost_tracker: Optional[BudgetChecker] = None) -> None:
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.github_token = config["core"].get("github_token", None)

    @abc.abstractmethod
    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        pass


class CodeSearchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__()

    def run(
        self, query: str, search_type: str = "repositories"
    ) -> Dict[str, Dict[str, str]]:
        print(f"[github API calling] Searching for code with query: {query}")
        results = {}

        try:
            idea = json.loads(query)
            if isinstance(idea, dict) and any(
                k in idea for k in ["Title", "Experiment"]
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
        import re

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
        keywords = []
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
            full_query = f"{' '.join(quoted_keywords[:max_terms//2])} {suffix}"

        return full_query

    def _search_github(
        self, query: str, search_type: str, result_limit: int = 10
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


class PaperSearchTool(BaseTool):
    def __init__(self, s2_api_key: Optional[str] = None, engine: Optional[str] = None, disable_fallback: bool = False) -> None:
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
            print(f"[INFO] Fallback to alternative search engine is DISABLED")
        
        if self.engine == "semanticscholar":
            if not self.disable_fallback:
                print("[INFO] Will fallback to OpenAlex if Semantic Scholar fails")
            if not self.s2_api_key:
                print("[INFO] No S2_API_KEY, rate limits will be stricter")
        elif self.engine == "openalex":
            mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
            if mail:
                print(f"[INFO] OpenAlex email configured: {mail}")
            else:
                print("[INFO] Recommend setting OPENALEX_MAIL_ADDRESS environment variable for better API access")
        
        print("[INFO] If you encounter search issues, OpenAlex is usually more stable than Semantic Scholar")

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        results = {}
        print(f"[PaperSearchTool] Searching for: {query}")
        papers = self.search_for_papers(query)

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
                if "openalex_id" in paper and paper.get("openalex_id"):
                    openalex_id = paper["openalex_id"]
                    bibtex = self._fetch_bibtex_from_openalex(openalex_id)
                    if bibtex:
                        paper_data["bibtex"] = bibtex
                        print(f"[PaperSearchTool] ✅ Got bibtex from OpenAlex")
                
                # Priority 2: Try Semantic Scholar (if it's S2 data and no OA bibtex)
                if not paper_data["bibtex"] and "paperId" in paper:
                    paper_id = paper.get("paperId")
                    if paper_id:
                        bibtex = self.fetch_bibtex(paper_id)
                        if bibtex and bibtex != "N/A":
                            paper_data["bibtex"] = bibtex
                            print(f"[PaperSearchTool] ✅ Got bibtex from Semantic Scholar")
                
                # Priority 3: Generate from metadata as fallback
                if not paper_data["bibtex"]:
                    bibtex = self._generate_bibtex_from_metadata(paper)
                    if bibtex:
                        paper_data["bibtex"] = bibtex
                        print(f"[PaperSearchTool] ⚠️ Generated bibtex from metadata (may have formatting issues)")
                    else:
                        print(f"[PaperSearchTool] ❌ No bibtex available")
                        continue  # Skip papers without bibtex
                
                # Try to enrich with OpenAlex if we got S2 data but want more info
                abstract_text = paper_data["abstract"] or ""
                if "paperId" in paper and len(abstract_text) < 100:
                    try:
                        print(f"[PaperSearchTool] Trying OpenAlex enrichment for short abstract...")
                        openalex_papers = self._search_openalex(paper_title, result_limit=1)
                        if openalex_papers and len(openalex_papers) > 0:
                            oa_paper = openalex_papers[0]
                            oa_abstract = oa_paper.get("abstract") or ""
                            if len(oa_abstract) > len(abstract_text):
                                paper_data["abstract"] = oa_abstract
                                paper_data["concepts"] = oa_paper.get("concepts", [])
                                paper_data["citationCount"] = oa_paper.get("citationCount", 0)
                                print(f"[PaperSearchTool] ✅ Enriched with OpenAlex data")
                    except Exception as e:
                        print(f"[PaperSearchTool] OpenAlex enrichment failed: {e}")
                
                results[paper_title] = paper_data
                abstract_len = len(paper_data.get('abstract', ''))
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
            print(f"(semantic scholar API calling) Searching for papers with query: {query}")
            try:
                result = self._search_semanticscholar(query, result_limit)
                if result:
                    return result
                else:
                    if self.disable_fallback:
                        print("[WARNING] Semantic Scholar returned no results. Fallback is disabled.")
                        return None
                    print("[INFO] Semantic Scholar returned no results, trying OpenAlex...")
            except Exception as e:
                if self.disable_fallback:
                    print(f"[ERROR] Semantic Scholar failed: {e}. Fallback is disabled.")
                    return None
                print(f"[WARNING] Semantic Scholar failed: {e}, trying OpenAlex as fallback...")

            # Fallback to OpenAlex (only if not disabled)
            try:
                print(f"(openalex API calling) Fallback search with query: {query}")
                return self._search_openalex(query, result_limit)
            except Exception as e:
                print(f"[ERROR] Both Semantic Scholar and OpenAlex failed: {e}")
                return None

        elif self.engine == "openalex":
            print(f"(openalex API calling) Searching for papers with query: {query}")
            try:
                result = self._search_openalex(query, result_limit)
                if result:
                    return result
                else:
                    if self.disable_fallback:
                        print("[WARNING] OpenAlex returned no results. Fallback is disabled.")
                        return None
                    print("[WARNING] OpenAlex returned no results, trying Semantic Scholar...")
            except Exception as e:
                if self.disable_fallback:
                    print(f"[ERROR] OpenAlex failed: {e}. Fallback is disabled.")
                    return None
                print(f"[WARNING] OpenAlex failed: {e}, trying Semantic Scholar as fallback...")
                
            # Fallback to Semantic Scholar (only if not disabled)
            try:
                print(f"(semantic scholar API calling) Fallback search with query: {query}")
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
                print(f"[Semantic Scholar] 403 Forbidden - API access denied")
                print(f"[INFO] This could be due to:")
                print(f"  1. Missing or invalid API key")
                print(f"  2. Rate limiting")
                print(f"  3. IP restrictions")
                if not self.s2_api_key:
                    print(f"[SUGGESTION] Get a free API key at: https://www.semanticscholar.org/product/api")
                # For 403 errors, don't retry, return None to allow fallback
                print(f"[Semantic Scholar] Skipping retries for 403 error, will try fallback engine")
                return None
            elif rsp.status_code == 429:
                print(f"[Semantic Scholar] 429 Rate Limited - will retry with exponential backoff")
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
    
    def _fetch_bibtex_from_openalex(self, work_id: str) -> Optional[str]:
        """Fetch BibTeX from OpenAlex by work ID (OpenAlex ID)"""
        try:
            import requests
            # OpenAlex provides direct bibtex endpoint
            # work_id should be like "W2741809807" or full URL
            if work_id.startswith("http"):
                bibtex_url = work_id.replace("https://openalex.org/", "https://api.openalex.org/") 
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
                    author_names = [a.get("name", "") for a in authors_raw if a.get("name")]
                    authors = " and ".join(author_names) if author_names else "Unknown Author"
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
                            author_names = [a.get("name", "") for a in parsed if isinstance(a, dict) and a.get("name")]
                            authors = " and ".join(author_names) if author_names else "Unknown Author"
                        else:
                            authors = "Unknown Author"
                    except:
                        authors = "Unknown Author"
                else:
                    authors = authors_raw
            else:
                authors = "Unknown Author"
            
            # Generate bibtex key (clean special characters)
            import re
            clean_title = re.sub(r'[^\w\s]', '', title)
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
                print(f"[Semantic Scholar] 403 Forbidden for bibtex fetch - API access denied")
                # For 403 errors, return N/A directly without retry
                return "N/A"
            elif rsp.status_code == 429:
                print(f"[Semantic Scholar] 429 Rate Limited for bibtex - will retry")
                raise
            else:
                print(f"[Semantic Scholar] HTTP Error {rsp.status_code} for bibtex: {e}")
                raise
        except requests.exceptions.RequestException as e:
            print(f"[Semantic Scholar] Bibtex fetch failed: {e}")
            raise
        except Exception as e:
            print(f"[Semantic Scholar] Unexpected bibtex error: {e}")
            raise

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
            top_concepts = sorted(work["concepts"], 
                                key=lambda x: x.get("score", 0), reverse=True)[:5]
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


class DrawerTool(BaseTool):
    def __init__(
        self,
        model: Any,
        prompt_template_dir: Optional[str] = None,
        temperature: float = 0.75,
    ):
        super().__init__()
        self.client, self.model = create_client(model)
        self.temperature = temperature

        # Load prompt templates using Config
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

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        try:
            query_dict = json.loads(query)
            section_name = query_dict.get("section_name")
            section_content = query_dict.get("section_content")
        except (json.JSONDecodeError, TypeError, AttributeError):
            raise ValueError(
                "Expected query to be a JSON string with 'section_name' and 'section_content'."
            )

        diagram = self.draw_diagram(
            section_name=section_name, section_content=section_content
        )

        results = {}
        if diagram:
            results["diagram"] = {
                "summary": diagram.get("summary", ""),
                "svg": diagram.get("svg", ""),
            }
        self.cost_tracker.report()
        return results

    def draw_diagram(
        self,
        section_name: str,
        section_content: str,
        msg_history: Optional[List[Dict[str, Any]]] = None,
        return_msg_history: bool = False,
    ) -> Any:
        # Use default system prompt if none provided
        section_prompt = self._get_section_prompts(section_name, section_content)

        diagram, updated_msg_history = self._generate_diagram(
            section_prompt, self.system_prompts, msg_history
        )

        return (diagram, updated_msg_history) if return_msg_history else diagram

    def _get_section_prompts(self, section_name: str, section_text: str) -> str:
        section_prompt = self.prompts.section_prompt[section_name].format(
            section_text=section_text
        )

        return section_prompt

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_diagram(
        self,
        section_prompt: str,
        drawer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Ensure msg_history is a list
        msg_history = msg_history or []

        # Generate diagram
        llm_response, msg_history = get_response_from_llm(
            section_prompt,
            model=self.model,
            client=self.client,
            system_message=drawer_system_prompt,
            msg_history=msg_history,
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="generate_diagram",
        )

        diagram = self._extract_diagram(llm_response)
        return diagram, msg_history

    def _extract_diagram(self, response: str) -> Dict[str, Any]:
        result = {"summary": "", "svg": "", "full_response": response}

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
        else:
            print("[ERROR] SVG missing or too short.")
        return result

    def _clean_svg(self, svg: str) -> str:
        # Strip any outer code block delimiters
        svg = svg.strip()
        svg = re.sub(r"^```(?:svg)?", "", svg)
        svg = re.sub(r"```$", "", svg)

        # Replace problematic ampersands
        svg = svg.replace("&", "&amp;")

        # Ensure no double XML declarations
        svg = re.sub(r"<\?xml.*?\?>", "", svg, count=1)

        # Remove extra whitespace lines
        svg = "\n".join([line for line in svg.splitlines() if line.strip()])

        return svg.strip()


class DockerExperimentRunner:
    def __init__(
        self,
        docker_image: str = "tiny-scientist-ml",
        docker_base: str = "python:3.11-slim",
    ):
        self.docker_image = docker_image
        self.docker_base = docker_base
        self.docker_client = None
        self.use_docker = False

        # Initialize Docker client
        try:
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

        # Common package name mappings (import_name -> pip_package_name)
        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "lxml": "lxml",
            "nltk": "nltk",
            "spacy": "spacy",
            "gensim": "gensim",
            "wordcloud": "wordcloud",
            "plotly": "plotly",
            "bokeh": "bokeh",
            "dash": "dash",
            "streamlit": "streamlit",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "sqlalchemy": "sqlalchemy",
            "psycopg2": "psycopg2-binary",
            "pymongo": "pymongo",
            "redis": "redis",
            "celery": "celery",
            "flask": "flask",
            "django": "django",
            "scipy": "scipy",
            "statsmodels": "statsmodels",
            "seaborn": "seaborn",
            "plotnine": "plotnine",
            "altair": "altair",
            "holoviews": "holoviews",
            "folium": "folium",
            "geopandas": "geopandas",
            "shapely": "shapely",
            "fiona": "fiona",
            "rasterio": "rasterio",
            "xarray": "xarray",
            "netcdf4": "netcdf4",
            "h5py": "h5py",
            "tables": "tables",
            "pyarrow": "pyarrow",
            "fastparquet": "fastparquet",
            "openpyxl": "openpyxl",
            "xlrd": "xlrd",
            "xlwt": "xlwt",
            "odf": "odfpy",
            "tabula": "tabula-py",
            "pdfplumber": "pdfplumber",
            "pymupdf": "pymupdf",
            "pypdf": "pypdf",
            "reportlab": "reportlab",
            "weasyprint": "weasyprint",
            "jinja2": "jinja2",
            "markdown": "markdown",
            "rst": "docutils",
            "sphinx": "sphinx",
            "mkdocs": "mkdocs",
            "jupyter": "jupyter",
            "ipython": "ipython",
            "notebook": "notebook",
            "jupyterlab": "jupyterlab",
            "voila": "voila",
            "nbconvert": "nbconvert",
            "papermill": "papermill",
            "nbdime": "nbdime",
            "nbstripout": "nbstripout",
            "jupytext": "jupytext",
        }

        # Standard library modules that don't need pip installation
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "random",
            "math",
            "statistics",
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
            "urllib",
            "http",
            "socket",
            "threading",
            "multiprocessing",
            "asyncio",
            "logging",
            "pickle",
            "copy",
            "typing",
            "abc",
            "enum",
            "dataclasses",
            "contextlib",
            "weakref",
            "gc",
            "inspect",
            "traceback",
            "pdb",
            "unittest",
            "doctest",
            "hashlib",
            "hmac",
            "base64",
            "zlib",
            "gzip",
            "bz2",
            "lzma",
            "zipfile",
            "tarfile",
            "csv",
            "configparser",
            "xml",
            "html",
            "email",
            "smtplib",
            "ftplib",
            "telnetlib",
            "poplib",
            "imaplib",
            "nntplib",
            "socketserver",
            "xmlrpc",
            "webbrowser",
            "cgi",
            "cgitb",
            "wsgiref",
            "urllib3",
            "ssl",
            "select",
            "signal",
            "pwd",
            "grp",
            "pwd",
            "spwd",
            "crypt",
            "termios",
            "tty",
            "pty",
            "fcntl",
            "pipes",
            "posix",
            "nt",
            "msvcrt",
            "winreg",
            "winsound",
            "msilib",
            "win32com",
            "win32api",
            "win32gui",
            "win32con",
            "win32file",
            "win32pipe",
            "win32event",
            "win32process",
            "win32security",
            "win32service",
            "win32serviceutil",
            "win32timezone",
            "pythoncom",
            "pywintypes",
            "win32ui",
            "win32print",
            "win32clipboard",
            "win32api",
            "win32gui",
            "win32con",
            "win32file",
            "win32pipe",
            "win32event",
            "win32process",
            "win32security",
            "win32service",
            "win32serviceutil",
            "win32timezone",
            "pythoncom",
            "pywintypes",
            "win32ui",
            "win32print",
            "win32clipboard",
        }

        try:
            with open(pyfile, "r") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            print(f"[Docker] Failed to parse {pyfile}: {e}")
            return []

        pkgs = set()
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

        # Map import names to pip package names and filter out base packages
        mapped_pkgs = []
        for pkg in pkgs:
            if pkg not in base_packages:
                pip_pkg = package_mapping.get(pkg, pkg)
                if pip_pkg not in mapped_pkgs:
                    mapped_pkgs.append(pip_pkg)

        return sorted(mapped_pkgs)

    def get_or_build_base_image(self) -> Optional[str]:
        """Build or get the base Docker image with common ML packages."""
        if not self.use_docker:
            return None
        if self.docker_client is not None:
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
        return None

    def get_or_build_experiment_image(self, experiment_py_path: str) -> Optional[str]:
        """Build or get experiment-specific Docker image with required packages."""
        if not self.use_docker:
            return None
        base_image = self.get_or_build_base_image()
        extra_pkgs = self.detect_required_packages(experiment_py_path)
        if extra_pkgs:
            image_name = f"tiny-scientist-exp-{hash(tuple(extra_pkgs))}"
            if self.docker_client is not None:
                try:
                    self.docker_client.images.get(image_name)
                    print(f"[Docker] Using cached experiment image: {image_name}")
                    return image_name
                except ImageNotFound:
                    print(
                        f"[Docker] Building experiment image: {image_name} with extra packages: {extra_pkgs}"
                    )
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Write requirements.txt
                        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                            for pkg in extra_pkgs:
                                f.write(pkg + "\n")
                        # Write Dockerfile with better error handling
                        dockerfile = f"""
FROM {base_image}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || (echo "Failed to install packages:" && cat requirements.txt && exit 1)
COPY experiment.py .
"""
                        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                            f.write(dockerfile)
                        # Copy experiment.py
                        shutil.copy(
                            experiment_py_path, os.path.join(tmpdir, "experiment.py")
                        )
                        try:
                            # Build with detailed logging
                            build_logs = self.docker_client.images.build(
                                path=tmpdir, tag=image_name, rm=True, decode=True
                            )
                            # Check for build errors
                            for log in build_logs:
                                if "error" in log:
                                    print(f"[Docker] Build error: {log['error']}")
                                    raise Exception(
                                        f"Docker build failed: {log['error']}"
                                    )
                                elif "stream" in log:
                                    print(f"[Docker] {log['stream'].strip()}")
                        except Exception as e:
                            print(f"[Docker] Failed to build image {image_name}: {e}")
                            # Fallback to base image
                            print(f"[Docker] Falling back to base image: {base_image}")
                            return base_image
                    return image_name
            return base_image
        else:
            return base_image

    def run_experiment_in_docker(
        self, experiment_code: str, run_num: int, output_dir: str, timeout: int = 7200
    ) -> Optional[Tuple[int, str]]:
        """Run experiment in a Docker container."""
        if not self.use_docker:
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy experiment.py
            exp_py = os.path.join(temp_dir, "experiment.py")
            with open(exp_py, "w") as f:
                f.write(experiment_code)

            # Detect and build experiment image
            image_name = self.get_or_build_experiment_image(exp_py)

            # Mount output dir for results
            output_dir = os.path.abspath(output_dir)

            container = None
            try:
                # Create container without auto-removal
                if self.docker_client is not None:
                    container = self.docker_client.containers.run(
                        image=image_name,
                        command=f"python experiment.py --out_dir=run_{run_num}",
                        volumes={
                            temp_dir: {"bind": "/experiment", "mode": "rw"},
                            output_dir: {"bind": "/experiment/output", "mode": "rw"},
                        },
                        working_dir="/experiment",
                        detach=True,
                        remove=False,  # Don't auto-remove
                        mem_limit="2g",
                        cpu_period=100000,
                        cpu_quota=50000,
                    )

                    print(f"[Docker] Container {container.id[:12]} started")

                    # Wait for container to finish
                    try:
                        result = container.wait(timeout=timeout)
                        print(
                            f"[Docker] Container {container.id[:12]} finished with status {result['StatusCode']}"
                        )
                    except Exception as wait_e:
                        print(f"[Docker] Container wait failed: {wait_e}")
                        # Try to get logs and stop container
                        try:
                            logs = container.logs().decode("utf-8")
                            container.stop(timeout=10)
                        except Exception:
                            logs = "Container failed to start or stopped unexpectedly"
                        return (
                            1,
                            f"Container execution failed: {wait_e}\nLogs: {logs}",
                        )

                    # Get logs before removing container
                    try:
                        logs = container.logs().decode("utf-8")
                    except Exception as log_e:
                        print(f"[Docker] Failed to get logs: {log_e}")
                        logs = "Failed to retrieve container logs"

                    if result["StatusCode"] == 0:
                        # Copy results from temp_dir/output/run_{run_num} to output_dir/run_{run_num}
                        src = os.path.join(temp_dir, f"run_{run_num}")
                        dst = os.path.join(output_dir, f"run_{run_num}")
                        if os.path.exists(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        return (0, logs)
                    else:
                        # Check if it's a missing package error
                        if "ModuleNotFoundError" in logs:
                            missing_pkg = self.extract_missing_package(logs)
                            print(f"[Docker] Missing package detected: {missing_pkg}")
                            # Try to install the missing package and retry
                            try:
                                # Create a new image with the missing package
                                retry_image_name = (
                                    f"tiny-scientist-retry-{hash(missing_pkg)}"
                                )
                                with tempfile.TemporaryDirectory() as retry_tmpdir:
                                    # Write requirements.txt with the missing package
                                    with open(
                                        os.path.join(retry_tmpdir, "requirements.txt"),
                                        "w",
                                    ) as f:
                                        f.write(missing_pkg + "\n")

                                    # Write Dockerfile
                                    retry_dockerfile = f"""
FROM {image_name}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
"""
                                    with open(
                                        os.path.join(retry_tmpdir, "Dockerfile"), "w"
                                    ) as f:
                                        f.write(retry_dockerfile)

                                    # Build retry image
                                    if self.docker_client is not None:
                                        self.docker_client.images.build(
                                            path=retry_tmpdir,
                                            tag=retry_image_name,
                                            rm=True,
                                        )

                                        # Run with retry image
                                        retry_container = self.docker_client.containers.run(
                                            image=retry_image_name,
                                            command=f"python experiment.py --out_dir=run_{run_num}",
                                            volumes={
                                                temp_dir: {
                                                    "bind": "/experiment",
                                                    "mode": "rw",
                                                },
                                                output_dir: {
                                                    "bind": "/experiment/output",
                                                    "mode": "rw",
                                                },
                                            },
                                            working_dir="/experiment",
                                            detach=True,
                                            remove=False,  # Don't auto-remove
                                            mem_limit="2g",
                                            cpu_period=100000,
                                            cpu_quota=50000,
                                        )

                                        print(
                                            f"[Docker] Retry container {retry_container.id[:12]} started"
                                        )
                                        retry_result = retry_container.wait(
                                            timeout=timeout
                                        )
                                        retry_logs = retry_container.logs().decode(
                                            "utf-8"
                                        )

                                        # Clean up retry container
                                        try:
                                            retry_container.remove(force=True)
                                        except Exception:
                                            pass

                                        if retry_result["StatusCode"] == 0:
                                            # Copy results
                                            src = os.path.join(
                                                temp_dir, f"run_{run_num}"
                                            )
                                            dst = os.path.join(
                                                output_dir, f"run_{run_num}"
                                            )
                                            if os.path.exists(src):
                                                shutil.copytree(
                                                    src, dst, dirs_exist_ok=True
                                                )
                                            return (0, retry_logs)
                                        else:
                                            return (
                                                retry_result["StatusCode"],
                                                retry_logs,
                                            )

                            except Exception as retry_e:
                                print(f"[Docker] Retry failed: {retry_e}")
                                return (result["StatusCode"], logs)

                        return (result["StatusCode"], logs)

            except Exception as e:
                print(f"[Docker] Experiment failed: {e}")
                return (1, f"Docker experiment failed: {e}")
            finally:
                # Clean up container
                if container:
                    try:
                        container.remove(force=True)
                        print(f"[Docker] Container {container.id[:12]} cleaned up")
                    except Exception as cleanup_e:
                        print(f"[Docker] Failed to cleanup container: {cleanup_e}")

        return (1, "Docker not available")

    def cleanup_docker_images(self) -> None:
        """Clean up Docker images created during experiments."""
        if not self.use_docker:
            return
        if self.docker_client is not None:
            try:
                images = self.docker_client.images.list()
                for image in images:
                    if image.tags and any(
                        "tiny-scientist" in tag for tag in image.tags
                    ):
                        print(f"[Docker] Removing image: {image.tags[0]}")
                        self.docker_client.images.remove(image.id, force=True)
            except Exception as e:
                print(f"[Docker] Failed to cleanup images: {e}")

    @staticmethod
    def extract_missing_package(stderr: str) -> str:
        """Extract missing package name from stderr with package name mapping."""

        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "lxml": "lxml",
            "nltk": "nltk",
            "spacy": "spacy",
            "gensim": "gensim",
            "wordcloud": "wordcloud",
            "plotly": "plotly",
            "bokeh": "bokeh",
            "dash": "dash",
            "streamlit": "streamlit",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "sqlalchemy": "sqlalchemy",
            "psycopg2": "psycopg2-binary",
            "pymongo": "pymongo",
            "redis": "redis",
            "celery": "celery",
            "flask": "flask",
            "django": "django",
            "scipy": "scipy",
            "statsmodels": "statsmodels",
            "seaborn": "seaborn",
            "plotnine": "plotnine",
            "altair": "altair",
            "holoviews": "holoviews",
            "folium": "folium",
            "geopandas": "geopandas",
            "shapely": "shapely",
            "fiona": "fiona",
            "rasterio": "rasterio",
            "xarray": "xarray",
            "netcdf4": "netcdf4",
            "h5py": "h5py",
            "tables": "tables",
            "pyarrow": "pyarrow",
            "fastparquet": "fastparquet",
            "openpyxl": "openpyxl",
            "xlrd": "xlrd",
            "xlwt": "xlwt",
            "odf": "odfpy",
            "tabula": "tabula-py",
            "pdfplumber": "pdfplumber",
            "pymupdf": "pymupdf",
            "pypdf": "pypdf",
            "reportlab": "reportlab",
            "weasyprint": "weasyprint",
            "jinja2": "jinja2",
            "markdown": "markdown",
            "rst": "docutils",
            "sphinx": "sphinx",
            "mkdocs": "mkdocs",
            "jupyter": "jupyter",
            "ipython": "ipython",
            "notebook": "notebook",
            "jupyterlab": "jupyterlab",
            "voila": "voila",
            "nbconvert": "nbconvert",
            "papermill": "papermill",
            "nbdime": "nbdime",
            "nbstripout": "nbstripout",
            "jupytext": "jupytext",
        }

        for line in stderr.splitlines():
            if "ModuleNotFoundError" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    import_name = parts[1]
                    # Return the mapped package name if it exists, otherwise return the original
                    return package_mapping.get(import_name, import_name)
        return "unknown-package"
