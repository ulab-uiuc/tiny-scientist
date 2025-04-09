import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .llm import extract_json_between_markers, get_response_from_llm
from .tool import PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff


class Thinker:
    def __init__(
            self,
            tools: List[Any],
            iter_num: int,
            model: str = "",
            client: Any = None,
            base_dir: str = "",
            config_dir: str = "",
            temperature: float = 0.75,
            s2_api_key: Optional[str] = None,
    ):
        self.tools = tools
        self.iter_num = iter_num
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.temperature = temperature
        self.searcher = PaperSearchTool()
        self.searcher.s2_api_key = s2_api_key

        # Load prompt templates
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "coder_prompt.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def think(self, intent: Dict[str, Any], check_novelty: bool,
              pdf_content: str) -> Dict[str, Dict[str, Any]]:
        """
        Generate a single research idea based on the provided intent.
        The intent may include an initial idea; if not, the intent itself is used.
        """
        # Use the "idea" field if available; otherwise, use intent directly.
        initial_ideas = [intent.get("idea", intent)]

        # Generate one idea
        new_ideas = self.generate_ideas(num_ideas=1, ideas=initial_ideas, num_reflections=self.iter_num,
                                        pdf_content=pdf_content)

        # Check novelty if requested
        if check_novelty and new_ideas:
            new_ideas = self.check_ideas(new_ideas, max_iterations=10)

        # Return the first idea or an empty dict
        if new_ideas:
            return {"idea": new_ideas[0]}
        else:
            return {"idea": {}}

    def rethink(self, info: Dict[str, Dict[str, Any]], current_round: int) -> Dict[str, Dict[str, Any]]:
        """
        Refine an existing research idea using one reflection iteration.
        """
        idea = info.get("idea", {})
        new_idea, _, _ = self._reflect_idea(
            idea,
            current_round=current_round,
            num_reflections=self.iter_num,
            msg_history=[]
        )
        return {"idea": new_idea} if new_idea else info

    def run(self, intent: Dict[str, Dict[str, str]], num_ideas: int = 1, check_novelty: bool = True,
            pdf_content: str = "") -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate and refine multiple research ideas based on the provided intent.
        """
        # 1. Check if we have an initial idea
        initial_idea = intent.get("idea", intent)

        all_ideas = []

        # Loop to generate and refine multiple ideas
        for i in range(num_ideas):
            print(f"\nProcessing idea {i + 1}/{num_ideas}")

            # 2. Generate a new idea using think
            new_idea_result = self.think({"idea": initial_idea}, check_novelty, pdf_content)
            if not new_idea_result["idea"]:
                print(f"Failed to generate idea {i + 1}")
                continue

            new_idea = new_idea_result["idea"]
            print(f"Generated idea: {new_idea.get('Name', 'Unnamed')}")

            # 3. Rethink the new idea iter_num times, using tools in each iteration
            current_idea = new_idea
            for j in range(self.iter_num):
                print(f"  Refinement iteration {j + 1}/{self.iter_num}")

                # Process through all tools
                for tool in self.tools:
                    tool_input = json.dumps(current_idea)
                    info = tool.run(tool_input)
                    current_idea.update(info)

                # Refine the idea
                refined = self.rethink({"idea": current_idea}, current_round=j + 1)
                current_idea = refined["idea"]

            all_ideas.append(current_idea)
            print(f"Completed refinement for idea: {current_idea.get('Name', 'Unnamed')}")

        # Save all ideas
        self.save_ideas(all_ideas)

        # Return all generated ideas
        return {"ideas": all_ideas}

    def generate_ideas(self, num_ideas: int = 1, ideas: Optional[List[Dict[str, Any]]] = None,
                       num_reflections: int = 5, pdf_content: str = "") -> List[Dict[str, Any]]:
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

        return idea_collection

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def generate_experiment_plan(self, idea: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    def check_ideas(self, ideas: List[Dict[str, Any]], max_iterations: int = 10,
                    engine: str = "semanticscholar") -> List[Dict[str, Any]]:
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

    def save_ideas(self, ideas: List[Dict[str, Any]]) -> None:
        output_path = osp.join(self.base_dir, "ideas.json")
        with open(output_path, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Saved {len(ideas)} ideas to {output_path}")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_idea(self, idea: Dict[str, Any], current_round: int, num_reflections: int,
                      msg_history: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
        # Ensure idea is a dict
        if isinstance(idea, list):
            idea = idea[0]
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
        if isinstance(new_idea, list):
            new_idea = new_idea[0]
        is_done = "I am done" in text

        if is_done:
            print(f"Idea refinement converged after {current_round} iterations.")

        return new_idea, msg_history, is_done

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_idea(self, idea_archive: List[Dict[str, Any]], num_reflections: int,
                       pdf_content: str) -> Optional[Dict[str, Any]]:
        # Ensure each entry in idea_archive is a dict (unwrap if necessary)
        idea_archive = [idea[0] if isinstance(idea, list) else idea for idea in idea_archive]

        # Format previous ideas
        prev_ideas_string = "\n\n".join(json.dumps(idea) for idea in idea_archive)

        pdf_section = f"Based on the content of the following paper:\n\n{pdf_content}\n\n" if pdf_content else ""

        # Search for related papers
        if idea_archive:
            last_idea = idea_archive[-1]
            last_idea_title = last_idea.get("Title", "")
            related_works_string = (
                self.searcher.search_for_papers(last_idea_title, result_limit=5)
                if last_idea_title else "No related works found."
            )
        else:
            related_works_string = "No related works found."

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

        idea = extract_json_between_markers(text)
        if isinstance(idea, list):
            idea = idea[0]

        if not idea:
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
    def _check_idea(self, idea: Dict[str, Any], max_iterations: int, engine: str) -> bool:
        msg_history: List[Dict[str, Any]] = []
        papers_str = ""
        for iteration in range(max_iterations):
            print(f"Novelty check iteration {iteration + 1}/{max_iterations}")

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
                print(f"Failed to get query in iteration {iteration + 1}")
                continue

            # Perform search
            query = json_output["Query"]
            print(f"Searching for: {query}")

            if not (papers := self.searcher.search_for_papers(query)):
                print(f"No papers found in iteration {iteration + 1}")
                continue

            papers_str = self.searcher.format_paper_results(papers)
            print(f"Found {len(papers)} relevant papers")

        print("Maximum iterations reached without decision, defaulting to not novel.")
        return False
