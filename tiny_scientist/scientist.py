import os
from typing import Any, Dict, List, Optional, Tuple, Union

import toml
from rich import print

from .budget_checker import BudgetChecker
from .coder import Coder
from .reviewer import Reviewer
from .safety_checker import SafetyChecker
from .thinker import Thinker
from .utils.input_formatter import InputFormatter
from .writer import Writer


class TinyScientist:
    MODULE_KEYS = (
        "safety_checker",
        "thinker",
        "coder",
        "writer",
        "reviewer",
    )
    DEFAULT_BUDGET_PREFERENCE = "balanced"
    BUDGET_WEIGHTS = {
        "balanced": {
            "safety_checker": 0.1,
            "thinker": 0.25,
            "writer": 0.25,
            "reviewer": 0.25,
            "coder": 0.15,
        },
        "write-heavy": {
            "safety_checker": 0.05,
            "thinker": 0.15,
            "writer": 0.5,
            "reviewer": 0.2,
            "coder": 0.1,
        },
        "think-heavy": {
            "safety_checker": 0.05,
            "thinker": 0.5,
            "writer": 0.15,
            "reviewer": 0.2,
            "coder": 0.1,
        },
        "review-heavy": {
            "safety_checker": 0.05,
            "thinker": 0.15,
            "writer": 0.15,
            "reviewer": 0.5,
            "coder": 0.15,
        },
    }

    @staticmethod
    def _coerce_budget(raw_budget: Optional[Union[float, int, str]]) -> Optional[float]:
        if raw_budget is None:
            return None
        try:
            budget_value = float(raw_budget)
        except (TypeError, ValueError) as exc:
            raise ValueError("Budget must be a number if provided.") from exc
        if budget_value < 0:
            raise ValueError("Budget must be non-negative.")
        return budget_value

    @classmethod
    def _normalize_budget_preference(cls, preference: Optional[str]) -> str:
        if not preference:
            preference = cls.DEFAULT_BUDGET_PREFERENCE
        normalized = preference.lower()
        if normalized not in cls.BUDGET_WEIGHTS:
            raise ValueError(f"Unknown budget preference: {preference}")
        return normalized

    @classmethod
    def _resolve_budget_inputs(
        cls,
        budget: Optional[Union[float, int, str]],
        budget_preference: Optional[str],
    ) -> Tuple[Optional[float], str]:
        normalized_budget = cls._coerce_budget(budget)
        normalized_preference = cls._normalize_budget_preference(budget_preference)
        return normalized_budget, normalized_preference

    @classmethod
    def resolve_budget_settings(
        cls,
        budget: Optional[Union[float, int, str]],
        budget_preference: Optional[str] = None,
    ) -> Tuple[Optional[float], str, Dict[str, Optional[float]]]:
        normalized_budget, normalized_preference = cls._resolve_budget_inputs(
            budget, budget_preference
        )
        if normalized_budget is None:
            allocation = {key: None for key in cls.MODULE_KEYS}
        else:
            weights = cls.BUDGET_WEIGHTS[normalized_preference]
            allocation = {
                key: normalized_budget * weights[key] for key in cls.MODULE_KEYS
            }
        return normalized_budget, normalized_preference, allocation

    @classmethod
    def compute_budget_alloca·tion(
        cls,
        budget: Optional[Union[float, int, str]],
        budget_preference: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        _, _, allocation = cls.resolve_budget_settings(budget, budget_preference)
        return allocation

    def __init__(
        self,
        model: str = "gpt-4o",
        output_dir: str = "./tiny_scientist_output",
        template: str = "acl",
        prompt_template_dir: Optional[str] = None,
        budget: Optional[float] = None,
        enable_safety_check: bool = True,
        budget_preference: Optional[str] = None,
        use_docker: bool = True,
    ):
        self.model = model
        self.output_dir = output_dir
        self.template = template
        self.prompt_template_dir = prompt_template_dir
        self.input_formatter = InputFormatter()
        self.enable_safety_check = enable_safety_check

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.toml"
        )
        config_core: Dict[str, Any] = {}
        if os.path.exists(config_path):
            cfg = toml.load(config_path)
            config_core = cfg.get("core", {})

        resolved_budget = budget
        if resolved_budget is None and "budget" in config_core:
            resolved_budget = config_core.get("budget")
        resolved_preference = budget_preference
        if resolved_preference is None:
            resolved_preference = config_core.get(
                "budget_preference", self.DEFAULT_BUDGET_PREFERENCE
            )

        (
            normalized_budget,
            normalized_preference,
            allocation,
        ) = self.resolve_budget_settings(resolved_budget, resolved_preference)

        self.cost = 0.0
        self.budget = normalized_budget
        self.budget_preference = normalized_preference
        self.global_cost_tracker = BudgetChecker(budget=normalized_budget)
        self.budget_allocation = allocation

        self.safety_checker = (
            SafetyChecker(
                model=model,
                cost_tracker=BudgetChecker(
                    budget=allocation.get("safety_checker"),
                    parent=self.global_cost_tracker,
                ),
            )
            if enable_safety_check
            else None
        )

        self.thinker = Thinker(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            iter_num=3,
            search_papers=True,
            generate_exp_plan=True,
            enable_safety_check=enable_safety_check,
            cost_tracker=BudgetChecker(
                budget=allocation.get("thinker"),
                parent=self.global_cost_tracker,
            ),
        )

        self.coder = Coder(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            max_iters=4,
            max_runs=3,
            cost_tracker=BudgetChecker(
                budget=allocation.get("coder"),
                parent=self.global_cost_tracker,
            ),
            use_docker=use_docker,
        )

        self.writer = Writer(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            template=template,
            cost_tracker=BudgetChecker(
                budget=allocation.get("writer"),
                parent=self.global_cost_tracker,
            ),
        )

        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            cost_tracker=BudgetChecker(
                budget=allocation.get("reviewer"),
                parent=self.global_cost_tracker,
            ),
        )

    def think(
        self, intent: str, num_ideas: int = 1, pdf_content: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if self.enable_safety_check and self.safety_checker:
            is_safe, safety_report = self.safety_checker.check_safety(intent)

            if not is_safe:
                print("❌ Safety check failed. Stopping execution.")
                print(f"Safety Report: {safety_report}")
                return {}

            print("✅ Safety check passed. Proceeding with idea generation...")

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

    def get_total_cost(self) -> float:
        """Get the total cost across all modules."""
        total_cost = 0.0
        modules = [
            ("Safety Checker", self.safety_checker),
            ("Thinker", self.thinker),
            ("Coder", self.coder),
            ("Writer", self.writer),
            ("Reviewer", self.reviewer),
        ]

        print("\n" + "=" * 50)
        print("📊 COST SUMMARY")
        print("=" * 50)

        for module_name, module in modules:
            if module and hasattr(module, "cost_tracker"):
                cost = module.cost_tracker.get_total_cost()
                total_cost += cost
                print(f"{module_name:15}: ${cost:.4f}")

        print("-" * 50)
        print(f"{'TOTAL COST':15}: ${total_cost:.4f}")
        print("=" * 50)

        return total_cost
