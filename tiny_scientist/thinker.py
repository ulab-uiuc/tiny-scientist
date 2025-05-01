import json
import os.path as osp
from typing import Any, Dict, List, Optional, Union, cast

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
        search_papers: bool = True,
        generate_exp_plan: bool = True,
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
        self.search_papers = search_papers
        self.generate_exp_plan = generate_exp_plan
        self.prompts = self.config.prompt_template.thinker_prompt
        self.intent = ""
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}

    def think(self, intent: str, pdf_content: Optional[str] = None) -> str:
        self.intent = intent
        print(f"Generating research idea based on: {intent}")

        pdf_content = self._load_pdf_content(pdf_content)
        if self.search_papers:
            query = self._generate_search_query(intent)
            related_works_string = self._get_related_works(query)
        else:
            related_works_string = "No Related Works Found"
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
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
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

            current_idea_exp = (
                self.generate_experiment_plan(current_idea_json)
                if self.generate_exp_plan
                else current_idea_json
            )

            current_idea_final = (
                self._check_novelty(current_idea_exp)
                if check_novelty
                else current_idea_exp
            )

            current_idea_dict = json.loads(current_idea_final)

            all_ideas.append(current_idea_dict)
            print(
                f"Completed refinement for idea: {current_idea_dict.get('Name', 'Unnamed')}"
            )
        if len(all_ideas) > 1:
            return all_ideas
        elif len(all_ideas) == 1:
            return cast(Dict[str, Any], all_ideas[0])
        else:
            print("No valid ideas generated.")
            return {}

    def rank(
        self, ideas: List[Dict[str, Any]], intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rank multiple research ideas."""
        intent = intent or self.intent

        ideas_json = json.dumps(ideas, indent=2)
        evaluation_result = self._get_idea_evaluation(ideas_json, intent)
        ranked_ideas = self._parse_evaluation_result(evaluation_result, ideas)

        return ranked_ideas

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def modify_idea(
        self,
        original_idea: Dict[str, Any],
        modifications: List[Dict[str, Any]],
        behind_idea: Optional[Dict[str, Any]] = None,
        all_ideas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Modify an idea based on score adjustments.
        """
        # Extract required information from modifications
        instruction_lines = []
        behind_content = (
            behind_idea.get("content", "") if behind_idea else "(No reference idea)"
        )

        for mod in modifications:
            metric_name = {
                "noveltyScore": "Novelty",
                "feasibilityScore": "Feasibility",
                "impactScore": "Impact",
            }.get(mod["metric"])

            direction = mod["direction"]
            instruction_lines.append(
                {
                    "metric": metric_name,
                    "direction": direction,
                    "reference": behind_content,
                }
            )

        # Prepare the prompt using the template from YAML
        prompt = self.prompts.modify_idea_prompt.format(
            idea=json.dumps(original_idea),
            modifications=json.dumps(instruction_lines),
            intent=self.intent,
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract modified idea from response
        modified_idea = extract_json_between_markers(text)
        if not modified_idea:
            print("Failed to extract modified idea")
            return original_idea

        # Apply metadata from original idea
        modified_idea["id"] = f"node-{len(all_ideas) + 1}" if all_ideas else "node-1"
        modified_idea["parent_id"] = original_idea.get("id", "unknown")
        modified_idea["is_modified"] = True

        # Re-rank the modified idea along with all other ideas
        if all_ideas:
            ranking_ideas = [
                idea for idea in all_ideas if idea.get("id") != original_idea.get("id")
            ]
            ranking_ideas.append(modified_idea)

            ranked_ideas = self.rank(ranking_ideas, self.intent)

            for idea in ranked_ideas:
                if idea.get("id") == modified_idea.get("id"):
                    return idea

        return modified_idea

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def merge_ideas(
        self,
        idea_a: Dict[str, Any],
        idea_b: Dict[str, Any],
        all_ideas: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge two ideas into a new one.
        """
        # Using the merge prompt template from YAML
        prompt = self.prompts.merge_ideas_prompt.format(
            idea_a=json.dumps(idea_a), idea_b=json.dumps(idea_b), intent=self.intent
        )

        # Call LLM to get merged content
        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        # Extract the merged idea from response
        merged_idea = extract_json_between_markers(text)
        if not merged_idea:
            print("Failed to extract merged idea")
            return None

        # Add metadata about the merged sources
        merged_idea["id"] = f"node-{len(all_ideas) + 1}" if all_ideas else "node-1"
        merged_idea["parent_ids"] = [
            idea_a.get("id", "unknown"),
            idea_b.get("id", "unknown"),
        ]
        merged_idea["is_merged"] = True

        # Re-rank the merged idea along with all other ideas
        if all_ideas:
            # Create a list with all ideas except the ones being merged, plus the new merged idea
            ranking_ideas = [
                idea
                for idea in all_ideas
                if idea.get("id") != idea_a.get("id")
                and idea.get("id") != idea_b.get("id")
            ]
            ranking_ideas.append(merged_idea)

            # Rank all ideas together
            ranked_ideas = self.rank(ranking_ideas, self.intent)

            # Find and return the merged idea from the ranked list
            for idea in ranked_ideas:
                if idea.get("id") == merged_idea.get("id"):
                    return idea

        # If no other ideas provided or ranking failed, return just the merged idea
        return merged_idea

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def generate_experiment_plan(self, idea: str) -> str:
        idea_dict = json.loads(idea)

        print("Generating experimental plan for the idea...")
        prompt = self.prompts.experiment_plan_prompt.format(
            idea=idea, intent=self.intent
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
            return idea

        idea_dict["Experiment"] = experiment_plan
        print("Experimental plan generated successfully.")

        return json.dumps(idea_dict, indent=2)

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

    def _get_idea_evaluation(self, ideas_json: str, intent: str) -> str:
        """Get comparative evaluation from LLM"""
        prompt = self.prompts.idea_evaluation_prompt.format(
            intent=intent, ideas=ideas_json
        )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.evaluation_system_prompt,
            msg_history=[],
            temperature=0.3,
        )

        return text

    def _parse_evaluation_result(
        self, evaluation_text: str, original_ideas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse evaluation result and update idea dictionaries with rankings"""
        # Extract JSON from response
        evaluation_data = extract_json_between_markers(evaluation_text)
        if not evaluation_data:
            print("Failed to extract JSON from evaluation response")
            return []
        # Create mapping from idea name to original idea dict
        idea_map = {idea.get("Name", ""): idea for idea in original_ideas}

        # Create ranked list
        ranked_ideas = []
        for ranked_item in evaluation_data.get("ranked_ideas", []):
            idea_name = ranked_item.get("Name", "")
            if idea_name in idea_map:
                # Get original idea and update with ranking data
                idea = idea_map[idea_name].copy()

                # Add ranking information
                idea["FeasibilityRanking"] = ranked_item.get("FeasibilityRanking")
                idea["NoveltyRanking"] = ranked_item.get("NoveltyRanking")
                idea["ImpactRanking"] = ranked_item.get("ImpactRanking")
                idea["NoveltyReason"] = ranked_item.get("NoveltyReason", "")
                idea["FeasibilityReason"] = ranked_item.get("FeasibilityReason", "")
                idea["ImpactReason"] = ranked_item.get("ImpactReason", "")
                # Remove all the scoring, using ranking instead
                if "Interestingness" in idea:
                    del idea["Interestingness"]
                if "Feasibility" in idea:
                    del idea["Feasibility"]
                if "Novelty" in idea:
                    del idea["Novelty"]
                if "IntentAlignment" in idea:
                    del idea["IntentAlignment"]
                if "Score" in idea:
                    del idea["Score"]
                ranked_ideas.append(idea)

        return ranked_ideas

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
