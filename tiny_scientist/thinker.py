import json
import os.path as osp
from typing import Any, Dict, List, Optional, Union, cast

from .configs import Config
from .tool import PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)


class Thinker:
    def __init__(
        self,
        tools: List[Any],
        iter_num: int,
        model: str = "",
        output_dir: str = "",
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
    ):
        self.tools = tools
        self.iter_num = iter_num
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.temperature = temperature
        self.config = Config(prompt_template_dir)
        self.searcher = PaperSearchTool()

        # Load prompt templates
        self.prompts = self.config.prompt_template.thinker_prompt
        self.found_papers: List[Dict[str, Any]] = []
        self.intent = ""

    def think(self, intent: str, pdf_content: Optional[str] = None) -> str:
        """
        Generate a single research idea based on the provided text intent.
        """
        self.intent = intent
        print(f"Generating research idea based on: {intent}")
        self.found_papers = []

        # Process PDF content if provided
        if pdf_content and osp.isfile(pdf_content):
            with open(pdf_content, "r", encoding="utf-8") as file:
                pdf_content = file.read()
            print(f"Using content from PDF file: {pdf_content}")

        related_works_string = self._search_and_add_references(intent, result_limit=5)

        # Generate the idea
        idea = self._generate_idea(intent, related_works_string, pdf_content)

        # Save the idea
        idea_dict = json.loads(idea)

        if self.found_papers:
            idea_dict["References"] = self.found_papers
        idea = json.dumps(idea_dict, indent=2)

        return idea

    def rethink(self, idea_json: str, current_round: int = 1) -> str:
        """
        Refine an existing research idea using one reflection iteration.
        """
        # Generate a search query for this idea
        query = self._generate_search_query(
            idea_json, intent=self.intent, query_type="rethink"
        )
        related_works = self.searcher.search_for_papers(query, result_limit=5)
        related_works_string = (
            self.searcher.format_paper_results(related_works)
            if related_works
            else "No related works found."
        )

        # Search for related papers
        refined_idea_json = self._reflect_idea(
            idea_json, current_round, related_works_string
        )

        return refined_idea_json

    def run(
        self,
        intent: str,
        num_ideas: int = 1,
        check_novelty: bool = False,
        pdf_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate and refine multiple research ideas based on the provided intent string.
        For each idea, perform iterative refinement and generate an experiment plan.
        """
        all_ideas = []

        # Process PDF content if provided as a file path
        if pdf_content and osp.isfile(pdf_content):
            with open(pdf_content, "r", encoding="utf-8") as file:
                pdf_content = file.read()
            print(f"Using content from PDF file: {pdf_content}")

        # Loop to generate and refine multiple ideas
        for i in range(num_ideas):
            print(f"\nProcessing idea {i + 1}/{num_ideas}")

            # Reset the papers collection for this new idea
            self.found_papers = []

            # Generate a new idea
            idea_json = self.think(intent, pdf_content)
            idea_dict = json.loads(idea_json)

            if not idea_dict:
                print(f"Failed to generate idea {i + 1}")
                continue

            print(f"Generated idea: {idea_dict.get('Name', 'Unnamed')}")

            # Rethink the new idea iter_num times, using tools in each iteration
            current_idea_json = idea_json
            for j in range(self.iter_num):
                print(f"Refining idea {j + 1}/{self.iter_num}")

                # Process through all tools
                current_idea_dict = json.loads(current_idea_json)
                for tool in self.tools:
                    tool_input = json.dumps(current_idea_dict)
                    info = tool.run(tool_input)
                    current_idea_dict.update(info)
                current_idea_json = json.dumps(current_idea_dict)

                # Refine the idea
                current_idea_json = self.rethink(current_idea_json, current_round=j + 1)

            # Generate an experimental plan for the idea after refinement
            current_idea_with_experiment = self._generate_experiment_plan(
                current_idea_json
            )

            # Check novelty if requested
            if check_novelty:
                current_idea_final = self._check_novelty(current_idea_with_experiment)
            else:
                current_idea_final = current_idea_with_experiment

            # Add the idea to our collection
            current_idea_dict = json.loads(current_idea_final)

            if self.found_papers:
                current_idea_dict["References"] = self.found_papers

            all_ideas.append(current_idea_dict)
            print(
                f"Completed refinement for idea: {current_idea_dict.get('Name', 'Unnamed')}"
            )

        if not all_ideas:
            print("No valid ideas generated.")
            return {}

        best_idea = cast(
            Dict[str, Any], max(all_ideas, key=lambda x: x.get("Score", 0))
        )
        return best_idea

    def _generate_search_query(
        self, content: str, intent: Optional[str] = None, query_type: str = "standard"
    ) -> str:
        """
        Generate an optimized search query based on the idea.
        """
        prompt = ""
        if query_type == "standard":
            prompt = self.prompts.query_prompt.format(intent=content)
        elif query_type == "rethink":
            prompt = self.prompts.rethink_query_prompt.format(
                intent=intent, idea=content
            )
        elif query_type == "novelty":
            prompt = self.prompts.novelty_query_prompt.format(
                intent=intent, idea=content
            )

        response, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract the query
        query_data = extract_json_between_markers(response)
        if query_data is None or "Query" not in query_data:
            return ""
        return str(query_data["Query"])

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_experiment_plan(self, idea_json: str) -> str:
        """
        Generate an experimental plan for the idea.
        """
        try:
            idea_dict = json.loads(idea_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            return idea_json

        print("Generating experimental plan for the idea...")

        # Get the prompt
        prompt = self.prompts.experiment_plan_prompt.format(
            idea=idea_json, intent=self.intent
        )

        # Call the LLM
        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract the experiment plan
        experiment_plan = extract_json_between_markers(text)
        if not experiment_plan:
            print("Failed to generate experimental plan.")
            return idea_json

        # Add the experiment plan to the idea
        idea_dict["Experiment"] = experiment_plan

        print("Experimental plan generated successfully.")
        return json.dumps(idea_dict, indent=2)

    def _save_ideas(self, ideas: List[str]) -> None:
        """
        Save ideas to a JSON file.
        """
        output_path = osp.join(self.output_dir, "ideas.json")
        with open(output_path, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Saved {len(ideas)} ideas to {output_path}")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_idea(
        self, idea_json: str, current_round: int, related_works_string: str
    ) -> str:
        """
        Refine an existing research idea.
        """
        # Get the prompt
        prompt = self.prompts.idea_reflection_prompt.format(
            intent=self.intent,
            current_round=current_round,
            num_reflections=self.iter_num,
            related_works_string=related_works_string,
        )

        # Call the LLM
        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract the refined idea
        new_idea = extract_json_between_markers(text)
        if isinstance(new_idea, list) and new_idea:
            new_idea = new_idea[0]

        # If no valid idea was extracted, return the original
        if not new_idea:
            print("Failed to extract a valid idea from refinement")
            return idea_json

        # Check if refinement is complete
        is_done = "I am done" in text
        if is_done:
            print(f"Idea refinement converged after {current_round} iterations.")

        return json.dumps(new_idea, indent=2)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_idea(
        self,
        intent: str,
        related_works_string: str,
        pdf_content: Optional[str] = None,
    ) -> str:
        """
        Generate a single research idea from an intent text.
        """
        # Format PDF content section if provided
        pdf_section = (
            f"Based on the content of the following paper:\n\n{pdf_content}\n\n"
            if pdf_content
            else ""
        )

        # Generate the idea
        text, _ = get_response_from_llm(
            self.prompts.idea_first_prompt.format(
                intent=intent,
                related_works_string=related_works_string,
                num_reflections=1,
                pdf_section=pdf_section,
            ),
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        idea = extract_json_between_markers(text)
        if isinstance(idea, list) and idea:
            idea = idea[0]

        if not idea:
            print("Failed to generate a valid idea")
            return json.dumps({})

        return json.dumps(idea, indent=2)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _check_novelty(self, idea_json: str, max_iterations: int = 10) -> str:
        """
        Check if the idea is novel by searching for similar papers.
        """
        try:
            idea_dict = json.loads(idea_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            return idea_json

        print(f"\nChecking novelty of idea: {idea_dict.get('Name', 'Unnamed')}")

        for iteration in range(max_iterations):
            print(f"Novelty check iteration {iteration + 1}/{max_iterations}")

            query = self._generate_search_query(
                idea_json, intent=self.intent, query_type="novelty"
            )
            papers = self.searcher.search_for_papers(query)

            if not papers:
                print(f"No papers found in iteration {iteration + 1}")
                papers_str = "No relevant papers found."
            else:
                papers_str = self.searcher.format_paper_results(papers)
                print(f"Found {len(papers)} relevant papers")

            if papers:
                for paper in papers:
                    self._add_reference(paper)

            # Get LLM decision or query
            prompt = self.prompts.novelty_prompt.format(
                current_round=iteration + 1,
                num_rounds=max_iterations,
                intent=self.intent,
                idea=idea_json,
                last_query_results=papers_str,
            )

            text, _ = get_response_from_llm(
                prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.novelty_system_prompt,
                msg_history=[],
            )

            # Process the decision
            if "NOVELTY CHECK: NOVEL" in text:
                print("Decision: Idea is novel")
                idea_dict["novel"] = True
                break
            elif "NOVELTY CHECK: NOT NOVEL" in text:
                print("Decision: Idea is not novel")
                idea_dict["novel"] = False
                break
            elif "NOVELTY CHECK: CONTINUE" in text:
                print("Decision: Need more information to determine novelty")
                continue
            else:
                print(f"No clear decision in iteration {iteration + 1}, continuing")

        # If no decision was made, default to not novel
        if "novel" not in idea_dict:
            print(
                "Maximum iterations reached without decision, defaulting to not novel."
            )
            idea_dict["novel"] = False

        return json.dumps(idea_dict, indent=2)

    def _format_authors(self, authors: Union[List[Any], str]) -> str:
        """
        Format author names according to our citation style.
        """
        if isinstance(authors, str):
            return authors

        if not isinstance(authors, list) or not authors:
            return "Unknown Authors"

        # Extract author names from list items
        author_names = []
        for author in authors:
            if isinstance(author, dict) and "name" in author:
                author_names.append(author["name"])
            elif isinstance(author, str):
                author_names.append(author)

        # If no valid names were found, return unknown
        if not author_names:
            return "Unknown Authors"

        # If there are more than three authors, use et al.
        if len(author_names) > 3:
            return f"{author_names[0]} et al."
        else:
            return ", ".join(author_names)

    def _add_reference(self, paper: Dict[str, Any]) -> None:
        """
        Add a paper to our references list, avoiding duplicates.
        """
        # Create a reference entry
        reference = {
            "id": f"ref{len(self.found_papers) + 1}",
            "title": paper.get("title", "Unknown Title"),
            "authors": self._format_authors(paper.get("authors", [])),
            "year": paper.get("year", "Unknown Year"),
        }

        # Check if we already have this paper by title
        for existing_paper in self.found_papers:
            if existing_paper["title"] == reference["title"]:
                return

        self.found_papers.append(reference)

    def _search_and_add_references(self, intent: str, result_limit: int = 5) -> str:
        """
        Search for papers and add them to our references.
        """
        query = self._generate_search_query(intent)
        papers = self.searcher.search_for_papers(query, result_limit=result_limit)
        if papers:
            for paper in papers:
                self._add_reference(paper)
        return (
            self.searcher.format_paper_results(papers)
            if papers
            else "No related works found."
        )
