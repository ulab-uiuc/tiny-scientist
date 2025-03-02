import json
import os.path as osp
from typing import Dict, List, Optional

import yaml

from .llm import extract_json_between_markers, get_response_from_llm
from .searcher import Searcher
from .utils.error_handler import api_calling_error_exponential_backoff


class Thinker:
    def __init__(
        self,
        model: str,
        client: any,
        base_dir: str,
        config_dir: str,
        temperature: float = 0.75,
        s2_api_key: Optional[str] = None
    ):
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.temperature = temperature
        self.searcher = Searcher(s2_api_key=s2_api_key)

        yaml_path = osp.join(config_dir, "thinker_prompt.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def generate_ideas(
        self,
        num_ideas: int = 1,
        ideas: List[Dict] = None,
        num_reflections: int = 5
    ) -> List[Dict]:
        if ideas is None:
            raise ValueError("Initial ideas must be provided")

        idea_collection = ideas.copy()
        original_size = len(idea_collection)

        print(f"Starting with {original_size} ideas, generating {num_ideas} new ideas")

        for i in range(num_ideas):
            idea_num = original_size + i + 1
            print(f"\nGenerating idea {idea_num}/{original_size + num_ideas}")

            new_idea = self._generate_idea(
                idea_collection,
                num_reflections
            )

            if new_idea:
                idea_collection.append(new_idea)
                print(f"Successfully generated idea: {new_idea.get('Name', 'Unnamed')}")
            else:
                print(f"Failed to generate idea {idea_num}")

        self.save_ideas(idea_collection)
        return idea_collection

    def check_ideas(
        self,
        ideas: List[Dict],
        max_iterations: int = 10,
        engine: str = "semanticscholar"
    ) -> List[Dict]:
        if ideas is None:
            raise ValueError("Ideas must be provided for novelty checking")

        for idx, idea in enumerate(ideas):
            if "novel" in idea:
                print(f"Skipping idea {idx}, already checked.")
                continue

            print(f"\nChecking novelty of idea {idx}: {idea['Name']}")
            novel = self._check_idea(idea, max_iterations, engine)
            idea["novel"] = novel

        self.save_ideas(ideas)
        return ideas

    def save_ideas(self, ideas: List[Dict]) -> None:
        with open(osp.join(self.base_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Saved {len(ideas)} ideas to {osp.join(self.base_dir, 'ideas.json')}")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_idea(
        self,
        idea_archive: List[Dict],
        num_reflections: int
    ) -> Optional[Dict]:
        # Format previous ideas
        idea_strings = [json.dumps(idea) for idea in idea_archive]
        prev_ideas_string = "\n\n".join(idea_strings)

        # Build prompt and initialize
        msg_history = []
        base_prompt = self.prompts["idea_first_prompt"].format(
            prev_ideas_string=prev_ideas_string,
            num_reflections=num_reflections,
        )

        # First generation step
        print(f"Iteration 1/{num_reflections}")
        text, msg_history = get_response_from_llm(
            base_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts["idea_system_prompt"],
            msg_history=msg_history,
            temperature=self.temperature,
        )

        idea = extract_json_between_markers(text)
        if not idea:
            return None

        # Search for related works
        query = idea.get('Title', '')
        if query:
            print(f"Searching for related works for idea: {query}")
            related_papers = self.searcher.search_for_papers(query)
            if related_papers:
                print(f"Found {len(related_papers)} related papers.")
                idea['RelatedWorks'] = self.searcher.format_paper_results(related_papers)
            else:
                print("No related works found.")

        # Reflection steps
        if num_reflections > 1:
            for j in range(num_reflections - 1):
                print(f"Iteration {j + 2}/{num_reflections}")

                text, msg_history = get_response_from_llm(
                    self.prompts["idea_reflection_prompt"].format(
                        current_round=j + 2,
                        num_reflections=num_reflections
                    ),
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts["idea_system_prompt"],
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

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _check_idea(
        self,
        idea: Dict,
        max_iterations: int,
        engine: str
    ) -> bool:
        msg_history = []
        papers_str = ""

        for iteration in range(max_iterations):
            print(f"Novelty check iteration {iteration+1}/{max_iterations}")

            # Get LLM decision or query
            text, msg_history = get_response_from_llm(
                self.prompts["novelty_prompt"].format(
                    current_round=iteration + 1,
                    num_rounds=max_iterations,
                    idea=idea,
                    last_query_results=papers_str,
                ),
                client=self.client,
                model=self.model,
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
            json_output = extract_json_between_markers(text)
            if not json_output or "Query" not in json_output:
                print(f"Failed to get query in iteration {iteration+1}")
                continue

            query = json_output["Query"]
            print(f"Searching for: {query}")

            # Perform search
            papers = self.searcher.search_for_papers(query, engine=engine)
            if not papers:
                print(f"No papers found in iteration {iteration+1}")
                continue

            papers_str = self.searcher.format_paper_results(papers)
            print(f"Found {len(papers)} relevant papers")

        print("Maximum iterations reached without decision, defaulting to not novel.")
        return False
