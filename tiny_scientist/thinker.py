import json
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, cast

from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .safety_checker import SafetyChecker
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
        cost_tracker: Optional[BudgetChecker] = None,
        enable_safety_check: bool = False,
        pre_reflection_threshold: float = 0.5,
        post_reflection_threshold: float = 0.8,
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

        # Enhanced criteria system from TinyScientistUI
        self.default_system_prompt = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You want to generate creative and impactful research ideas that can be feasibly investigated with the code provided.
Be critical and realistic in your assessments."""

        self.default_novelty_criteria = (
            "How original is the idea compared to existing work?"
        )
        self.default_feasibility_criteria = (
            "How practical is implementation within reasonable resource constraints?"
        )
        self.default_impact_criteria = "What is the potential impact of this research on the field and broader applications?"

        # Initialize with defaults
        self.system_prompt = self.default_system_prompt
        self.novelty_criteria = self.default_novelty_criteria
        self.feasibility_criteria = self.default_feasibility_criteria
        self.impact_criteria = self.default_impact_criteria

        # Legacy criteria descriptions for backward compatibility
        self.default_criteria_descriptions = """1. Intent Alignment: How well does each idea address the original research intent?
        2. Scientific Merit: How significant is the potential contribution to the field?
        3. Novelty: How original is the idea compared to existing work?
        4. Feasibility: How practical is implementation within reasonable resource constraints?
        5. Impact: What is the potential impact of this research on the field and broader applications?"""
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.pre_reflection_threshold = pre_reflection_threshold
        self.post_reflection_threshold = post_reflection_threshold

        self.enable_safety_check = enable_safety_check
        self.safety_checker: Optional[SafetyChecker]
        if self.enable_safety_check:
            self.safety_checker = SafetyChecker(
                model=self.model, cost_tracker=self.cost_tracker
            )
        else:
            self.safety_checker = None

    def think(self, intent: str, pdf_content: Optional[str] = None) -> str:
        self.intent = intent
        # If intent is too long, show simplified message
        if len(intent) > 100:
            print("Generating children ideas")
        else:
            print(f"Generating research idea based on: {intent}")

        pdf_content = self._load_pdf_content(pdf_content)
        if self.search_papers:
            query = self._generate_search_query(intent)
            related_works_string = self._get_related_works(query)
        else:
            related_works_string = "No Related Works Found"
        idea = self._generate_idea(intent, related_works_string, pdf_content)

        self.cost_tracker.report()
        return idea

    def rethink(self, idea_json: str, current_round: int = 1) -> str:
        print(f"Rethinking idea in round {current_round}...")
        if self.search_papers:
            query = self._generate_search_query(
                idea_json, intent=self.intent, query_type="rethink"
            )
            related_works_string = self._get_related_works(query)
        else:
            related_works_string = "No Related Works Found"

        self.cost_tracker.report()
        return self._reflect_idea(idea_json, current_round, related_works_string)

    def _process_single_idea(
        self,
        intent: str,
        pdf_content: Optional[str],
        idea_index: int,
        total_ideas: int,
        check_novelty: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Process a single idea through the complete pipeline."""
        print(f"\nProcessing idea {idea_index + 1}/{total_ideas}")

        idea_json = self.think(intent, pdf_content)
        idea_dict = json.loads(idea_json)

        if not idea_dict:
            print(f"Failed to generate idea {idea_index + 1}")
            return None

        print(f"Generated idea: {idea_dict.get('Title', 'Unnamed')}")

        current_idea_json = self._refine_idea(idea_json)

        current_idea_exp = (
            self.generate_experiment_plan(current_idea_json)
            if self.generate_exp_plan
            else current_idea_json
        )

        current_idea_final = (
            self._check_novelty(current_idea_exp) if check_novelty else current_idea_exp
        )

        # Apply comprehensive safety check if enabled
        current_idea_final = self._safety_check(current_idea_final)

        current_idea_dict = json.loads(current_idea_final)

        print(
            f"Completed refinement for idea: {current_idea_dict.get('Name', 'Unnamed')}"
        )
        return current_idea_dict

    def run(
        self,
        intent: str,
        num_ideas: int = 1,
        check_novelty: bool = False,
        pdf_content: Optional[str] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        all_ideas = []
        pdf_content = self._load_pdf_content(pdf_content)

        if num_ideas == 1:
            # For single idea, use sequential processing to avoid thread overhead
            idea_dict = self._process_single_idea(
                intent, pdf_content, 0, 1, check_novelty
            )
            if idea_dict:
                all_ideas.append(idea_dict)
        else:
            # For multiple ideas, use parallel processing
            print(f"🚀 Starting parallel generation of {num_ideas} ideas...")

            with ThreadPoolExecutor(max_workers=min(num_ideas, 3)) as executor:
                # Submit all idea generation tasks
                future_to_index = {
                    executor.submit(
                        self._process_single_idea,
                        intent,
                        pdf_content,
                        i,
                        num_ideas,
                        check_novelty,
                    ): i
                    for i in range(num_ideas)
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idea_index = future_to_index[future]
                    try:
                        result = future.result()
                        if result:
                            all_ideas.append(result)
                    except Exception as e:
                        print(f"Error processing idea {idea_index + 1}: {str(e)}")
                        continue

            print(
                f"✅ Parallel generation completed. Generated {len(all_ideas)} out of {num_ideas} ideas."
            )

        if len(all_ideas) > 1:
            self.cost_tracker.report()
            return all_ideas
        elif len(all_ideas) == 1:
            self.cost_tracker.report()
            return cast(Dict[str, Any], all_ideas[0])
        else:
            print("No valid ideas generated.")
            self.cost_tracker.report()
            return {}

    def get_system_prompt(self) -> str:
        """Get the current system prompt"""
        return self.system_prompt

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """Set the system prompt, use None to reset to default"""
        if prompt is None:
            self.system_prompt = self.default_system_prompt
        else:
            self.system_prompt = prompt

    def get_criteria(self, dimension: str) -> str:
        """Get criteria for a specific dimension"""
        criteria_map = {
            "novelty": self.novelty_criteria,
            "feasibility": self.feasibility_criteria,
            "impact": self.impact_criteria,
        }
        return criteria_map.get(dimension, "")

    def set_criteria(self, dimension: str, criteria: Optional[str] = None) -> None:
        """Set criteria for a specific dimension. If no criteria is provided, reset to default"""
        if dimension == "novelty":
            self.novelty_criteria = (
                criteria if criteria else self.default_novelty_criteria
            )
        elif dimension == "feasibility":
            self.feasibility_criteria = (
                criteria if criteria else self.default_feasibility_criteria
            )
        elif dimension == "impact":
            self.impact_criteria = (
                criteria if criteria else self.default_impact_criteria
            )
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

    def rank(
        self,
        ideas: List[Dict[str, Any]],
        intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rank multiple research ideas."""
        intent = intent or self.intent

        ideas_json = json.dumps(ideas, indent=2)
        evaluation_result = self._get_idea_evaluation(ideas_json, intent)
        ranked_ideas = self._parse_evaluation_result(evaluation_result, ideas)

        self.cost_tracker.report()
        return ranked_ideas

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def modify_idea(
        self,
        original_idea: Dict[str, Any],
        modifications: List[Dict[str, Any]],
        behind_idea: Optional[Dict[str, Any]] = None,
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
            cost_tracker=self.cost_tracker,
            task_name="modify_idea",
        )

        # Extract modified idea from response
        modified_idea = extract_json_between_markers(text)
        if not modified_idea:
            print("Failed to extract modified idea")
            return original_idea

        self.cost_tracker.report()
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
            cost_tracker=self.cost_tracker,
            task_name="merge_ideas",
        )

        # Extract the merged idea from response
        merged_idea = extract_json_between_markers(text)
        if not merged_idea:
            print("Failed to extract merged idea")
            return None

        # If no other ideas provided or ranking failed, return just the merged idea
        self.cost_tracker.report()
        return merged_idea

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def generate_experiment_plan(self, idea: str) -> str:
        idea_dict = json.loads(idea)
        is_experimental = idea_dict.get("is_experimental", True)

        print("Generating experimental plan for the idea...")
        if is_experimental:
            prompt = self.prompts.experiment_plan_prompt.format(
                idea=idea, intent=self.intent
            )
        else:
            prompt = self.prompts.non_experiment_plan_prompt.format(
                idea=idea, intent=self.intent
            )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.idea_system_prompt,
            msg_history=[],
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="generate_experiment_plan",
        )

        # Extract both the original JSON and the new Markdown table
        experiment_plan_json = extract_json_between_markers(text)
        try:
            experiment_plan_table = text.split("```markdown")[1].split("```")[0].strip()
        except IndexError:
            experiment_plan_table = None

        if not experiment_plan_json or not experiment_plan_table:
            print("Failed to generate one or both parts of the experimental plan.")
            # Return the original idea if generation fails
            return idea

        # Store the JSON in 'Experiment' and the table in 'ExperimentTable'
        idea_dict["Experiment"] = experiment_plan_json
        idea_dict["ExperimentTable"] = experiment_plan_table
        print("Dual-format experimental plan generated successfully.")

        self.cost_tracker.report()
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

        # Skip reflections entirely if budget usage is already high
        budget = self.cost_tracker.get_budget()
        if (
            budget is not None
            and self.cost_tracker.get_total_cost() / budget
            >= self.pre_reflection_threshold
        ):
            print("[Thinker] Skipping idea reflections due to budget limit.")
            self.cost_tracker.report()
            return current_idea_json

        max_rounds = self.iter_num
        rounds_done = 0
        per_round_cost = None

        while rounds_done < max_rounds:
            print(f"Refining idea {rounds_done + 1}th time out of {max_rounds} times.")

            start_cost = self.cost_tracker.get_total_cost()
            current_idea_dict = json.loads(current_idea_json)
            for tool in self.tools:
                tool_input = json.dumps(current_idea_dict)
                info = tool.run(tool_input)
                current_idea_dict.update(info)
            current_idea_json = json.dumps(current_idea_dict)

            current_idea_json = self.rethink(
                current_idea_json, current_round=rounds_done + 1
            )

            iteration_cost = self.cost_tracker.get_total_cost() - start_cost
            if per_round_cost is None:
                per_round_cost = max(iteration_cost, 1e-6)
                if budget is not None:
                    allowed = budget * self.post_reflection_threshold
                    remaining = allowed - self.cost_tracker.get_total_cost()
                    additional = int(max(0.0, remaining) // per_round_cost)
                    max_rounds = min(self.iter_num, 1 + additional)

            rounds_done += 1
            if (
                budget is not None
                and self.cost_tracker.get_total_cost()
                >= budget * self.post_reflection_threshold
            ):
                break

        self.cost_tracker.report()
        return current_idea_json

    def _get_idea_evaluation(
        self, ideas_json: str, intent: str, custom_criteria: Optional[str] = None
    ) -> str:
        """Get comparative evaluation from LLM"""
        prompt = self.prompts.idea_evaluation_prompt.format(
            intent=intent,
            ideas=ideas_json,
            novelty_criteria=self.novelty_criteria,
            feasibility_criteria=self.feasibility_criteria,
            impact_criteria=self.impact_criteria,
        )
        if custom_criteria:
            prompt = prompt.replace(self.default_criteria_descriptions, custom_criteria)

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.evaluation_system_prompt,
            msg_history=[],
            temperature=0.3,
            cost_tracker=self.cost_tracker,
            task_name="get_idea_evaluation",
        )

        self.cost_tracker.report()
        return text

    def _parse_evaluation_result(
        self, evaluation_text: str, original_ideas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse evaluation result and update idea dictionaries with scores"""
        # Extract JSON from response
        evaluation_data = extract_json_between_markers(evaluation_text)

        if not evaluation_data:
            print("Failed to extract JSON from evaluation response")
            return []
        # Create mapping from idea title to original idea dict (check both Title and title)
        idea_map = {}
        for idea in original_ideas:
            title = idea.get("Title", "") or idea.get("title", "")
            if title:
                idea_map[title] = idea

        # Create scored list
        scored_ideas = []
        scored_items = evaluation_data.get("scored_ideas", [])

        # FIX: The key from the prompt is "scored_ideas", not "ranked_ideas"
        for scored_item in scored_items:
            idea_name = scored_item.get("Title", "")

            if idea_name in idea_map:
                # Get original idea and update with scoring data
                idea = idea_map[idea_name].copy()

                # Add scoring information
                idea["FeasibilityScore"] = scored_item.get("FeasibilityScore")
                idea["NoveltyScore"] = scored_item.get("NoveltyScore")
                idea["ImpactScore"] = scored_item.get("ImpactScore")
                idea["NoveltyReason"] = scored_item.get("NoveltyReason", "")
                idea["FeasibilityReason"] = scored_item.get("FeasibilityReason", "")
                idea["ImpactReason"] = scored_item.get("ImpactReason", "")

                scored_ideas.append(idea)

        self.cost_tracker.report()
        return scored_ideas

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

        self.cost_tracker.report()
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
            cost_tracker=self.cost_tracker,
            task_name="generate_search_query",
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
            cost_tracker=self.cost_tracker,
            task_name="reflect_idea",
        )

        new_idea = extract_json_between_markers(text)
        if isinstance(new_idea, list) and new_idea:
            new_idea = new_idea[0]

        if not new_idea:
            print("Failed to extract a valid idea from refinement")
            return idea_json

        if "I am done" in text:
            print(f"Idea refinement converged after {current_round} iterations.")

        self.cost_tracker.report()
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
            cost_tracker=self.cost_tracker,
            task_name="generate_idea",
        )

        idea = extract_json_between_markers(text)
        if isinstance(idea, list) and idea:
            idea = idea[0]

        if not idea:
            print("Failed to generate a valid idea")
            return json.dumps({})

        # Extract comparison table if present
        try:
            comparison_table = text.split("```markdown")[1].split("```")[0].strip()
            idea["ComparisonTable"] = comparison_table
        except IndexError:
            # No comparison table found, continue without it
            pass

        self.cost_tracker.report()
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
                cost_tracker=self.cost_tracker,
                task_name="check_novelty",
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

        self.cost_tracker.report()
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

    def _safety_check(self, idea_json: str) -> str:
        """
        Check and enhance the safety of a research idea.

        Args:
            idea_json: JSON string containing the research idea

        Returns:
            str: Modified idea JSON with enhanced safety
        """
        if not self.enable_safety_check or not self.safety_checker:
            return idea_json

        print("🔒 Applying comprehensive safety check...")

        try:
            # Parse the idea JSON
            idea_dict = json.loads(idea_json)

            # Use the integrated SafetyChecker for comprehensive safety evaluation
            safety_result = self.safety_checker.comprehensive_safety_check(
                self.intent, idea_dict
            )

            # Check if the idea passed all safety checks
            if safety_result["overall_safety"]["is_safe"]:
                # If there's an enhanced idea from ethics evaluation, use it
                if safety_result.get("idea_ethics") and safety_result[
                    "idea_ethics"
                ].get("ethics_evaluation", {}).get("enhanced_idea"):
                    enhanced_idea = safety_result["idea_ethics"]["ethics_evaluation"][
                        "enhanced_idea"
                    ]
                    print("✅ Safety check passed with enhancements")
                    return json.dumps(enhanced_idea, indent=2)
                else:
                    print("✅ Safety check passed")
                    return idea_json
            else:
                print(
                    f"⚠️ Safety check warning: {safety_result['overall_safety']['recommendation']}"
                )
                return idea_json

        except json.JSONDecodeError:
            print("⚠️ Safety check failed to parse idea JSON, using original")
            return idea_json
        except Exception as e:
            print(f"⚠️ Safety check error: {str(e)}, using original idea")
            return idea_json
