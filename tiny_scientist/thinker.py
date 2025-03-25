import json
import os.path as osp
from typing import Dict, List, Optional, Tuple

import yaml

from .llm import extract_json_between_markers, get_response_from_llm
from .tool import PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff


class Thinker:
    def __init__(self, model: str, client: any, base_dir: str, config_dir: str,
                 temperature: float = 0.75, s2_api_key: Optional[str] = None):
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.temperature = temperature
        self.searcher = PaperSearchTool(s2_api_key=s2_api_key)

        # Load prompt templates
        with open(osp.join(config_dir, "thinker_prompt.yaml"), "r") as f:
            self.prompts = yaml.safe_load(f)

    def generate_ideas(self, num_ideas: int = 1, ideas: List[Dict] = None,
                       num_reflections: int = 5, pdf_content: str = "") -> List[Dict]:
        if not ideas:
            raise ValueError("Initial ideas must be provided")

        idea_collection = ideas.copy()
        original_size = len(idea_collection)
        print(f"Starting with {original_size} ideas, generating {num_ideas} new ideas")

        for i in range(num_ideas):
            print(f"\nGenerating idea {original_size + i + 1}/{original_size + num_ideas}")

            new_idea = self._generate_idea(idea_collection, num_reflections, pdf_content)
            if new_idea:
                # Generate an experimental plan for the newly generated idea.
                experiment_plan = self.generate_experiment_plan(new_idea)
                if experiment_plan:
                    new_idea["ExperimentPlan"] = experiment_plan

                query = new_idea.get("Title", "")
                searched_papers = self.searcher.search_for_papers(query=query, result_limit=5)
                simplified_papers = self.searcher.simplify_papers(searched_papers) if searched_papers else []
                new_idea["SearchedPapers"] = simplified_papers if simplified_papers else []

                # Add the cited papers
                citation_queries = new_idea.get("CitationQueries", [])
                aggregated_papers = []
                for query in citation_queries:
                    papers = self.searcher.search_for_papers(query=query, result_limit=3)
                    if papers:
                        aggregated_papers.extend(self.searcher.simplify_papers(papers))

                seen_titles = set()
                final_papers = []
                for paper in aggregated_papers:
                    if paper["title"] not in seen_titles:
                        final_papers.append(paper)
                        seen_titles.add(paper["title"])
                simplified_final_papers = self.searcher.simplify_papers(final_papers) if final_papers else []
                new_idea["CitedPapers"] = simplified_final_papers if simplified_final_papers else []

                idea_collection.append(new_idea)
                print(f"Successfully generated idea: {new_idea.get('Name', 'Unnamed')}")
            else:
                print(f"Failed to generate idea {original_size + i + 1}")
        self.save_ideas(idea_collection)
        return idea_collection

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def generate_experiment_plan(self, idea: Dict) -> Optional[Dict]:
        print("Generating experimental plan for the idea...")
        experiment_text, experiment_msg_history = get_response_from_llm(
            self.prompts["experiment_plan_prompt"].format(idea=json.dumps(idea, indent=2)),
            client=self.client, model=self.model,
            system_message=self.prompts["idea_system_prompt"],
            msg_history=[], temperature=self.temperature,
        )
        experiment_plan = extract_json_between_markers(experiment_text)
        if experiment_plan:
            print("Experimental plan generated successfully.")
        else:
            print("Failed to generate experimental plan.")
        return experiment_plan

    def check_ideas(self, ideas: List[Dict], max_iterations: int = 10,
                   engine: str = "semanticscholar") -> List[Dict]:
        if not ideas:
            raise ValueError("Ideas must be provided for novelty checking")

        for idx, idea in enumerate(ideas):
            if "novel" in idea:
                print(f"Skipping idea {idx}, already checked.")
                continue

            print(f"\nChecking novelty of idea {idx}: {idea['Name']}")
            idea["novel"] = self._check_idea(idea, max_iterations, engine)

        self.save_ideas(ideas)
        return ideas

    def save_ideas(self, ideas: List[Dict]) -> None:
        output_path = osp.join(self.base_dir, "ideas.json")
        with open(output_path, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Saved {len(ideas)} ideas to {output_path}")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_idea(self, idea: Dict, current_round: int, num_reflections: int,
                     msg_history: List[Dict]) -> Tuple[Optional[Dict], List[Dict], bool]:
        print(f"Iteration {current_round}/{num_reflections}")

        related_works_string = self.searcher.search_for_papers(idea.get("Title", ""), result_limit=5)

        text, msg_history = get_response_from_llm(
            self.prompts["idea_reflection_prompt"].format(
                current_round=current_round,
                num_reflections=num_reflections,
                related_works_string=related_works_string
        ),
            client=self.client, model=self.model,
            system_message=self.prompts["idea_system_prompt"],
            msg_history=msg_history, temperature=self.temperature,
        )

        new_idea = extract_json_between_markers(text)
        is_done = "I am done" in text

        if is_done:
            print(f"Idea refinement converged after {current_round} iterations.")

        return new_idea, msg_history, is_done

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_idea(self, idea_archive: List[Dict], num_reflections: int, pdf_content: str) -> Optional[Dict]:
        # Format previous ideas
        prev_ideas_string = "\n\n".join(json.dumps(idea) for idea in idea_archive)

        pdf_section = ""

        if pdf_content:
            pdf_section = f"Based on the content of the following paper:\n\n{pdf_content}\n\n"

        # Search for related papers
        if idea_archive:
            # Use the title of the most recent idea as a search query
            last_idea_title = idea_archive[-1].get("Title", "")
            if last_idea_title:
                related_works_string = self.searcher.search_for_papers(last_idea_title, result_limit=5)
            else:
                related_works_string = "No related works found."
        else:
            related_works_string = "No related works found."

        # First generation step
        print(f"Iteration 1/{num_reflections}")
        text, msg_history = get_response_from_llm(
            self.prompts["idea_first_prompt"].format(
                prev_ideas_string=prev_ideas_string,
                related_works_string=related_works_string,
                num_reflections=num_reflections,
                pdf_section=pdf_section
            ),
            client=self.client, model=self.model,
            system_message=self.prompts["idea_system_prompt"],
            msg_history=[], temperature=self.temperature,
        )

        if not (idea := extract_json_between_markers(text)):
            return None

        # Reflection steps
        if num_reflections > 1:
            for j in range(num_reflections - 1):
                current_round = j + 2

                new_idea, msg_history, is_done = self._reflect_idea(
                    idea=idea,
                    current_round=current_round,
                    num_reflections=num_reflections,
                    msg_history=msg_history
                )

                if not new_idea:
                    break

                idea = new_idea

                if is_done:
                    break

        return idea

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _check_idea(self, idea: Dict, max_iterations: int, engine: str) -> bool:
        msg_history = []
        papers_str = ""

        for iteration in range(max_iterations):
            print(f"Novelty check iteration {iteration+1}/{max_iterations}")

            # Get LLM decision or query
            text, msg_history = get_response_from_llm(
                self.prompts["novelty_prompt"].format(
                    current_round=iteration + 1, num_rounds=max_iterations,
                    idea=idea, last_query_results=papers_str,
                ),
                client=self.client, model=self.model,
                system_message=self.prompts["novelty_system_prompt"],
                msg_history=msg_history,
            )

            # Check for termination conditions
            if "decision made: novel" in text.lower():
                print("Decision: Idea is novel")
                return True
            if "decision made: not novel" in text.lower():
                print("Decision: Idea is not novel")
                return False

            # Extract and process search query
            if not (json_output := extract_json_between_markers(text)) or "Query" not in json_output:
                print(f"Failed to get query in iteration {iteration+1}")
                continue

            # Perform search
            query = json_output["Query"]
            print(f"Searching for: {query}")

            if not (papers := self.searcher.search_for_papers(query, engine=engine)):
                print(f"No papers found in iteration {iteration+1}")
                continue

            papers_str = self.searcher.format_paper_results(papers)
            print(f"Found {len(papers)} relevant papers")

        print("Maximum iterations reached without decision, defaulting to not novel.")
        return False
