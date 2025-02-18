import json
import os
import os.path as osp
import time
from typing import Dict, List, Optional

import backoff
import requests
import yaml

from .llm import extract_json_between_markers, get_response_from_llm


class Thinker:
    def __init__(
        self,
        model: str,
        client: any,
        base_dir: str,
        temperature: float = 0.75,
        s2_api_key: Optional[str] = None
    ):
        """Initialize the Thinker with model configuration and prompt templates."""
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.temperature = temperature
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")

        # Load prompt templates
        yaml_path = os.path.join(os.path.dirname(__file__), "thinker.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        # Load experiment code and task description
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            self.code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt_data = json.load(f)
            self.task_description = prompt_data["task_description"]
            self.system_prompt = prompt_data.get("system")

    def generate_ideas(
        self,
        skip_generation: bool = False,
        max_num_generations: int = 20,
        num_reflections: int = 5,
    ) -> List[Dict]:
        """Generate new research ideas based on the experiment code."""
        if skip_generation:
            return self._load_existing_ideas()

        idea_archive = self._load_seed_ideas()

        for i in range(max_num_generations):
            print(f"\nGenerating idea {i + 1}/{max_num_generations}")
            try:
                new_idea = self._generate_single_idea(idea_archive, num_reflections)
                if new_idea:
                    idea_archive.append(new_idea)
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

        self._save_ideas(idea_archive)
        return idea_archive

    def generate_next_idea(
        self,
        prev_idea_archive: List[Dict] = [],
        num_reflections: int = 5,
        max_attempts: int = 10,
    ) -> List[Dict]:
        """Generate the next research idea based on previous ideas."""
        idea_archive = prev_idea_archive.copy()
        original_archive_size = len(idea_archive)

        print(f"Generating idea {original_archive_size + 1}")

        if not idea_archive:
            idea_archive.extend(self._load_seed_ideas()[:1])
        else:
            for _ in range(max_attempts):
                try:
                    new_idea = self._generate_single_idea(
                        idea_archive,
                        num_reflections,
                        include_scores=True
                    )
                    if new_idea:
                        idea_archive.append(new_idea)
                        break
                except Exception as e:
                    print(f"Failed to generate idea: {e}")
                    continue

        self._save_ideas(idea_archive)
        return idea_archive

    def check_idea_novelty(
        self,
        ideas: List[Dict],
        max_num_iterations: int = 10,
        engine: str = "semanticscholar"
    ) -> List[Dict]:
        """Check the novelty of generated ideas against existing literature."""
        for idx, idea in enumerate(ideas):
            if "novel" in idea:
                print(f"Skipping idea {idx}, already checked.")
                continue

            print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

            novel = False
            msg_history = []
            papers_str = ""

            for j in range(max_num_iterations):
                try:
                    decision, query = self._process_novelty_iteration(
                        idea, papers_str, j + 1, max_num_iterations, msg_history
                    )

                    if decision is not None:
                        novel = decision
                        break

                    papers = self._search_for_papers(query, engine=engine)
                    papers_str = self._format_paper_results(papers)

                except Exception as e:
                    print(f"Error: {e}")
                    continue

            idea["novel"] = novel

        self._save_ideas(ideas)
        return ideas

    def _generate_single_idea(
        self,
        idea_archive: List[Dict],
        num_reflections: int,
        include_scores: bool = False
    ) -> Optional[Dict]:
        """Generate a single new idea with potential reflections."""
        idea_strings = [json.dumps(idea) for idea in idea_archive]
        prev_ideas_string = "\n\n".join(idea_strings)

        msg_history = []
        base_prompt = self.prompts["idea_first_prompt"].format(
            task_description=self.task_description,
            code=self.code,
            prev_ideas_string=prev_ideas_string,
            num_reflections=num_reflections,
        )

        if include_scores:
            base_prompt += """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
"""

        print(f"Iteration 1/{num_reflections}")
        text, msg_history = get_response_from_llm(
            base_prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt or self.prompts["idea_system_prompt"],
            msg_history=msg_history,
            temperature=self.temperature,
        )

        idea = extract_json_between_markers(text)
        if not idea:
            return None

        if num_reflections > 1:
            idea = self._reflect_on_idea(idea, num_reflections, msg_history)

        return idea

    def _reflect_on_idea(
        self,
        idea: Dict,
        num_reflections: int,
        msg_history: List[Dict]
    ) -> Dict:
        """Perform reflection iterations on an idea."""
        for j in range(num_reflections - 1):
            print(f"Iteration {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                self.prompts["idea_reflection_prompt"].format(
                    current_round=j + 2,
                    num_reflections=num_reflections
                ),
                client=self.client,
                model=self.model,
                system_message=self.system_prompt or self.prompts["idea_system_prompt"],
                msg_history=msg_history,
                temperature=self.temperature,
            )

            new_idea = extract_json_between_markers(text)
            if not new_idea:
                break

            idea = new_idea
            if "I am done" in text:
                print(f"Idea generation converged after {j + 2} iterations.")
                break

        return idea

    def _process_novelty_iteration(
        self,
        idea: Dict,
        papers_str: str,
        current_round: int,
        max_rounds: int,
        msg_history: List[Dict]
    ) -> tuple[Optional[bool], Optional[str]]:
        """Process a single iteration of novelty checking."""
        text, msg_history = get_response_from_llm(
            self.prompts["novelty_prompt"].format(
                current_round=current_round,
                num_rounds=max_rounds,
                idea=idea,
                last_query_results=papers_str,
            ),
            client=self.client,
            model=self.model,
            system_message=self.prompts["novelty_system_prompt"].format(
                num_rounds=max_rounds,
                task_description=self.task_description,
                code=self.code,
            ),
            msg_history=msg_history,
        )

        if "decision made: novel" in text.lower():
            return True, None
        if "decision made: not novel" in text.lower():
            return False, None

        json_output = extract_json_between_markers(text)
        return None, json_output["Query"] if json_output else None

    @staticmethod
    def _format_paper_results(papers: Optional[List[Dict]]) -> str:
        """Format paper results into a string."""
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

    def _load_existing_ideas(self) -> List[Dict]:
        """Load existing ideas from file."""
        try:
            with open(osp.join(self.base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
            return []
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")
            return []

    def _load_seed_ideas(self) -> List[Dict]:
        """Load seed ideas from file."""
        with open(osp.join(self.base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        return [json.loads(json.dumps(idea)) for idea in seed_ideas]

    def _save_ideas(self, ideas: List[Dict]) -> None:
        """Save ideas to file."""
        with open(osp.join(self.base_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.HTTPError,
        on_backoff=self._on_backoff
    )
    def _search_for_papers(
        self,
        query: str,
        result_limit: int = 10,
        engine: str = "semanticscholar"
    ) -> Optional[List[Dict]]:
        """Search for papers using the specified search engine."""
        if not query:
            return None

        if engine == "semanticscholar":
            return self._search_semanticscholar(query, result_limit)
        elif engine == "openalex":
            return self._search_openalex(query, result_limit)
        else:
            raise NotImplementedError(f"{engine=} not supported!")

    def _search_semanticscholar(
        self,
        query: str,
        result_limit: int
    ) -> Optional[List[Dict]]:
        """Search papers using Semantic Scholar API."""
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
        if not results["total"]:
            return None

        time.sleep(1.0)
        return results["data"]

    def _search_openalex(
        self,
        query: str,
        result_limit: int
    ) -> Optional[List[Dict]]:
        """Search papers using OpenAlex API."""
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

        papers = [self._extract_work_info(work) for work in works]
        return papers

    @staticmethod
    def _extract_work_info(work: any, max_abstract_length: int = 1000) -> Dict[str, str]:
        """Extract relevant information from an OpenAlex work object."""
        # Find venue
        venue = "Unknown"
        for location in work["locations"]:
            if location["source"] is not None:
                potential_venue = location["source"]["display_name"]
                if potential_venue:
                    venue = potential_venue
                    break

        # Get authors
        authors_list = [
            author["author"]["display_name"]
            for author in work["authorships"]
        ]
        authors = (
            " and ".join(authors_list)
            if len(authors_list) < 20
            else f"{authors_list[0]} et al."
        )

        # Get and truncate abstract if needed
        abstract = work["abstract"] or ""
        if len(abstract) > max_abstract_length:
            print(
                f"[WARNING] {work['title']=}: {len(abstract)=} is too long! "
                f"Use first {max_abstract_length} chars."
            )
            abstract = abstract[:max_abstract_length]

        return {
            "title": work["title"],
            "authors": authors,
            "venue": venue,
            "year": work["publication_year"],
            "abstract": abstract,
            "citationCount": work["cited_by_count"],
        }

    @staticmethod
    def _on_backoff(details: Dict) -> None:
        """Callback for backoff decorator."""
        print(
            f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
            f"calling function {details['target'].__name__} at {time.strftime('%X')}"
        )
