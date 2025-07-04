import json
from typing import List, Optional

from rich import print

from .configs import Config
from .tool import BaseTool
from .utils.cost_tracker import CostTracker
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)


class Planner:
    def __init__(
        self,
        tools: List[BaseTool] = None,
        model: str = "gpt-4o",
        output_dir: str = "./",
        prompt_template_dir: Optional[str] = None,
        temperature: float = 0.7,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.tools = tools or []
        self.output_dir = output_dir
        self.temperature = temperature
        # Initialize LLM client
        self.client, self.model = create_client(model)

        self.config = Config(prompt_template_dir)
        self.prompts = self.config.prompt_template.planner_prompt
        self.cost_tracker = cost_tracker or CostTracker()

        self.intent = ""

    def run(self, idea: str, intent: str = "") -> str:
        """Main entry point for generating experiment plans."""
        self.intent = intent
        return self.generate_experiment_plan(idea)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def generate_experiment_plan(self, idea: str) -> str:
        """Generate experiment plan for a given research idea."""
        try:
            idea_dict = json.loads(idea)
        except json.JSONDecodeError as e:
            print(f"Error parsing idea JSON: {e}")
            return idea

        is_experimental = idea_dict.get("is_experimental", True)

        print("Generating experimental plan for the idea...")
        if is_experimental:
            print("Generating experimental plan for AI-related idea...")
            prompt = self.prompts.experiment_plan_prompt.format(
                idea=idea, intent=self.intent
            )
        else:
            print("Generating research plan for non-experimental idea...")
            prompt = self.prompts.non_experiment_plan_prompt.format(
                idea=idea, intent=self.intent
            )

        text, _ = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.planner_system_prompt,
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

        if self.cost_tracker:
            self.cost_tracker.report()
        return json.dumps(idea_dict, indent=2)
