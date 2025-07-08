import os
from typing import Any, Dict, List, Optional, Tuple, Union

import toml

from rich import print

from .coder import Coder
from .reviewer import Reviewer
from .thinker import Thinker
from .utils.checker import Checker
from .utils.input_formatter import InputFormatter
from .writer import Writer


class TinyScientist:
    def __init__(
        self,
        model: str = "gpt-4o",
        output_dir: str = "./",
        template: str = "acl",
        prompt_template_dir: Optional[str] = None,
        budget: Optional[float] = None,
        budget_preference: Optional[str] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.template = template
        self.prompt_template_dir = prompt_template_dir
        self.input_formatter = InputFormatter()

        self.cost = 0.0

        if budget_preference is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
            if os.path.exists(config_path):
                cfg = toml.load(config_path)
                budget_preference = cfg.get("core", {}).get("budget_preference", "balanced")
            else:
                budget_preference = "balanced"

        weights = {
            "balanced": {"thinker": 0.3, "writer": 0.3, "reviewer": 0.3, "coder": 0.1},
            "write-heavy": {"thinker": 0.2, "writer": 0.5, "reviewer": 0.2, "coder": 0.1},
            "think-heavy": {"thinker": 0.5, "writer": 0.2, "reviewer": 0.2, "coder": 0.1},
            "review-heavy": {"thinker": 0.2, "writer": 0.2, "reviewer": 0.5, "coder": 0.1},
        }
        if budget_preference not in weights:
            raise ValueError(f"Unknown budget preference: {budget_preference}")

        allocation = {
            k: (budget * w if budget is not None else None)
            for k, w in weights[budget_preference].items()
        }

        self.thinker = Thinker(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            iter_num=3,
            search_papers=True,
            generate_exp_plan=True,
            enable_ethical_defense=False,
            cost_tracker=Checker(budget=allocation.get("thinker")),
        )

        self.coder = Coder(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            max_iters=4,
            max_runs=3,
            cost_tracker=Checker(budget=allocation.get("coder")),
        )

        self.writer = Writer(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            template=template,
            cost_tracker=Checker(budget=allocation.get("writer")),
        )

        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            cost_tracker=Checker(budget=allocation.get("reviewer")),
        )

    def think(
        self, intent: str, num_ideas: int = 1, pdf_content: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        print("🧠 Generating idea...")
        ideas = self.thinker.run(
            intent=intent, num_ideas=num_ideas, pdf_content=pdf_content
        )
        print(ideas)
        print("✅ Idea generated.")
        return ideas

    def code(
        self,
        idea: Dict[str, Any],
        baseline_results: Optional[Dict[str, Any]] = {},
    ) -> Tuple[bool, str]:
        print("💻 Running experiments...")
        status, exp_path, error_details = self.coder.run(
            idea=idea, baseline_results=baseline_results
        )
        if status:
            print(f"✅ Experiment completed successfully. Results saved at {exp_path}")
        else:
            print(f"❌ Experiment failed. Please check {exp_path} for details.")
            if error_details:
                print(f"Error details: {error_details}")
        return status, exp_path

    def write(self, idea: Dict[str, Any], experiment_dir: str) -> str:
        print("📝 Writing paper...")
        pdf_path, paper_name = self.writer.run(idea=idea, experiment_dir=experiment_dir)
        print(
            f"Check the generated paper named as {paper_name} and saved at {pdf_path}"
        )
        print("✅ Paper written.")
        return pdf_path

    def review(self, pdf_path: str) -> Dict[str, Any]:
        print("🔍 Reviewing paper...")
        review = self.reviewer.run(pdf_path=pdf_path)
        print(review)
        print("✅ Review complete.")
        return review
