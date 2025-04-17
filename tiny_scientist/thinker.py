import json
import os.path as osp
from typing import Any, Dict, List, Optional, cast

from rich import print

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
        self.prompts = self.config.prompt_template.thinker_prompt
        self.intent = ""
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}

    def think(self, intent: str, pdf_content: Optional[str] = None) -> str:
        self.intent = intent
        print(f"Generating research idea based on: {intent}")

        pdf_content = self._load_pdf_content(pdf_content)
        query = self._generate_search_query(intent)
        related_works_string = self._get_related_works(query)

        idea = self._generate_idea(intent, related_works_string, pdf_content)

        return idea

    def rethink(self, idea_json: str, current_round: int = 1) -> str:
        query = self._generate_search_query(
            idea_json, intent=self.intent, query_type="rethink"
        )
        related_works_string = self._get_related_works(query)

        return self._reflect_idea(idea_json, current_round, related_works_string)

    def run(
        self,
        intent: str,
        num_ideas: int = 1,
        check_novelty: bool = False,
        pdf_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        all_ideas = []
        pdf_content = self._load_pdf_content(pdf_content)

        for i in range(num_ideas):
            print(f"\nProcessing idea {i + 1}/{num_ideas}")

            idea_json = self.think(intent, pdf_content)
            idea_dict = json.loads(idea_json)

            if not idea_dict:
                print(f"Failed to generate idea {i + 1}")
                continue

            print(f"Generated idea: {idea_dict.get('Title', 'Unnamed')}")

            current_idea_json = self._refine_idea(idea_json)
            current_idea_with_experiment = self._generate_experiment_plan(
                current_idea_json
            )

            current_idea_final = (
                self._check_novelty(current_idea_with_experiment)
                if check_novelty
                else current_idea_with_experiment
            )

            current_idea_dict = json.loads(current_idea_final)

            all_ideas.append(current_idea_dict)
            print(
                f"Completed refinement for idea: {current_idea_dict.get('Name', 'Unnamed')}"
            )

        if not all_ideas:
            print("No valid ideas generated.")
            return {}

        return cast(Dict[str, Any], max(all_ideas, key=lambda x: x.get("Score", 0)))

    def _load_pdf_content(self, pdf_path: Optional[str] = None) -> Optional[str]:
        if pdf_path and osp.isfile(pdf_path):
            with open(pdf_path, "r", encoding="utf-8") as file:
                content = file.read()
            print(f"Using content from PDF file: {pdf_path}")
            return content
        return None

    def _refine_idea(self, idea_json: str) -> str:
        current_idea_json = idea_json

        for j in range(self.iter_num):
            print(f"Refining idea {j + 1}th time out of {self.iter_num} times.")

            current_idea_dict = json.loads(current_idea_json)
            for tool in self.tools:
                tool_input = json.dumps(current_idea_dict)
                info = tool.run(tool_input)
                current_idea_dict.update(info)
            current_idea_json = json.dumps(current_idea_dict)

            current_idea_json = self.rethink(current_idea_json, current_round=j + 1)

        return current_idea_json

    def _get_related_works(self, query: str) -> str:
        """Get related works using query caching, similar to Reviewer class"""
        if query in self._query_cache:
            related_papers = self._query_cache[query]
            print("✅ Using cached query results")
        else:
            print(f"Searching for papers with query: {query}")
            results_dict = self.searcher.run(query)
            related_papers = list(results_dict.values()) if results_dict else []
            self._query_cache[query] = related_papers

            if related_papers:
                print("✅ Related Works Found")
            else:
                print("❎ No Related Works Found")

        return self._format_paper_results(related_papers)

    def _generate_search_query(
        self, content: str, intent: Optional[str] = None, query_type: str = "standard"
    ) -> str:
        prompt_mapping = {
            "standard": self.prompts.query_prompt.format(intent=content),
            "rethink": self.prompts.rethink_query_prompt.format(
                intent=intent, idea=content
            ),
            "novelty": self.prompts.novelty_query_prompt.format(
                intent=intent, idea=content
            ),
        }

        prompt = prompt_mapping.get(query_type, "")
        response, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        query_data = extract_json_between_markers(response)
        return str(query_data.get("Query", "")) if query_data else ""

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_experiment_plan(self, idea_json: str) -> str:
        try:
            idea_dict = json.loads(idea_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            return idea_json

        print("Generating experimental plan for the idea...")
        prompt = self.prompts.experiment_plan_prompt.format(
            idea=idea_json, intent=self.intent
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        experiment_plan = extract_json_between_markers(text)
        if not experiment_plan:
            print("Failed to generate experimental plan.")
            return idea_json

        idea_dict["Experiment"] = experiment_plan
        print("Experimental plan generated successfully.")

        return json.dumps(idea_dict, indent=2)

    def _save_ideas(self, ideas: List[str]) -> None:
        output_path = osp.join(self.output_dir, "ideas.json")
        with open(output_path, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Saved {len(ideas)} ideas to {output_path}")

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_idea(
        self, idea_json: str, current_round: int, related_works_string: str
    ) -> str:
        prompt = self.prompts.idea_reflection_prompt.format(
            intent=self.intent,
            current_round=current_round,
            num_reflections=self.iter_num,
            related_works_string=related_works_string,
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        new_idea = extract_json_between_markers(text)
        if isinstance(new_idea, list) and new_idea:
            new_idea = new_idea[0]

        if not new_idea:
            print("Failed to extract a valid idea from refinement")
            return idea_json

        if "I am done" in text:
            print(f"Idea refinement converged after {current_round} iterations.")

        return json.dumps(new_idea, indent=2)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_idea(
        self,
        intent: str,
        related_works_string: str,
        pdf_content: Optional[str] = None,
    ) -> str:
        pdf_section = (
            f"Based on the content of the following paper:\n\n{pdf_content}\n\n"
            if pdf_content
            else ""
        )

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
            papers_str = self._get_related_works(query)

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

        if "novel" not in idea_dict:
            print(
                "Maximum iterations reached without decision, defaulting to not novel."
            )
            idea_dict["novel"] = False

        return json.dumps(idea_dict, indent=2)

    @staticmethod
    def _format_paper_results(papers: List[Dict[str, Any]]) -> str:
        """Format paper results exactly like Reviewer class"""
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"{i}: {paper.get('title', 'No title')}. {paper.get('source', 'No authors')}. "
                f"{paper.get('info', 'No venue')}"
            )

        return "\n\n".join(paper_strings)
