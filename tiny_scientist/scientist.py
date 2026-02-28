import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import toml
from rich import print

from .budget_checker import BudgetChecker
from .coder import Coder
from .reviewer import Reviewer
from .safety_checker import SafetyChecker
from .thinker import Thinker
from .utils.agent_sdk import (
    AgentSdk,
    resolve_agent_sdk,
    validate_agent_sdk_model_combo,
)
from .utils.input_formatter import InputFormatter
from .utils.pricing import BUDGET_MODULE_KEYS as PRICING_BUDGET_MODULE_KEYS
from .utils.pricing import (
    DEFAULT_BUDGET_PREFERENCE as PRICING_DEFAULT_BUDGET_PREFERENCE,
)
from .utils.pricing import (
    compute_budget_allocation as pricing_compute_budget_allocation,
)
from .utils.pricing import resolve_budget_settings as pricing_resolve_budget_settings
from .utils.rich_output import (
    print_mapping_table,
    print_rows_table,
    summarize_idea,
    summarize_review,
)
from .writer import Writer


class TinyScientist:
    MODULE_KEYS = PRICING_BUDGET_MODULE_KEYS
    DEFAULT_BUDGET_PREFERENCE = PRICING_DEFAULT_BUDGET_PREFERENCE

    @classmethod
    def resolve_budget_settings(
        cls,
        budget: Optional[Union[float, int, str]],
        budget_preference: Optional[str] = None,
    ) -> Tuple[Optional[float], str, Dict[str, Optional[float]]]:
        return pricing_resolve_budget_settings(budget, budget_preference)

    @classmethod
    def compute_budget_allocation(
        cls,
        budget: Optional[Union[float, int, str]],
        budget_preference: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        return pricing_compute_budget_allocation(budget, budget_preference)

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        output_dir: str = "./tiny_scientist_output",
        template: str = "acl",
        prompt_template_dir: Optional[str] = None,
        budget: Optional[float] = None,
        enable_safety_check: bool = True,
        thinker_iter_num: int = 3,
        budget_preference: Optional[str] = None,
        use_docker: bool = True,
        agent_sdk: Optional[str] = None,
        use_claude_agent_sdk: Optional[bool] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.template = template
        self.prompt_template_dir = prompt_template_dir
        self.input_formatter = InputFormatter()
        self.enable_safety_check = enable_safety_check
        self.agent_sdk: AgentSdk = resolve_agent_sdk(
            agent_sdk=agent_sdk,
            use_claude_agent_sdk=use_claude_agent_sdk,
        )
        validate_agent_sdk_model_combo(self.agent_sdk, model)

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.toml"
        )
        config_core: Dict[str, Any] = {}
        if os.path.exists(config_path):
            cfg = toml.load(config_path)
            config_core = cfg.get("core", {})

        resolved_budget = budget if budget is not None else config_core.get("budget")
        resolved_preference = (
            budget_preference
            if budget_preference is not None
            else config_core.get("budget_preference")
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
            iter_num=thinker_iter_num,
            search_papers=True,
            generate_exp_plan=True,
            enable_safety_check=enable_safety_check,
            cost_tracker=BudgetChecker(
                budget=allocation.get("thinker"),
                parent=self.global_cost_tracker,
            ),
            agent_sdk=self.agent_sdk,
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
            agent_sdk=self.agent_sdk,
        )
        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            cost_tracker=BudgetChecker(
                budget=allocation.get("reviewer"),
                parent=self.global_cost_tracker,
            ),
            agent_sdk=self.agent_sdk,
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
            agent_sdk=self.agent_sdk,
        )

    def think(
        self,
        intent: str,
        num_ideas: int = 1,
        pdf_content: Optional[str] = None,
        check_novelty: bool = False,
        save_json: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if self.enable_safety_check and self.safety_checker:
            is_safe, safety_report = self.safety_checker.check_safety(intent)

            if not is_safe:
                print("❌ Safety check failed. Stopping execution.")
                print_mapping_table("Safety Failure", safety_report)
                return {}

            print("✅ Safety check passed. Proceeding with idea generation...")

        print("🧠 Generating idea...")
        ideas = self.thinker.run(
            intent=intent,
            num_ideas=num_ideas,
            pdf_content=pdf_content,
            check_novelty=check_novelty,
        )
        if isinstance(ideas, dict) and ideas:
            print_mapping_table("Idea Summary", summarize_idea(ideas))
        elif isinstance(ideas, list):
            rows = []
            for idx, idea in enumerate(ideas, start=1):
                if isinstance(idea, dict):
                    rows.append(
                        {
                            "index": idx,
                            "title": idea.get("Title") or idea.get("Name"),
                            "metric": idea.get("Metric"),
                            "citations": (
                                len(idea.get("Citations", []))
                                if isinstance(idea.get("Citations"), list)
                                else idea.get("Citations")
                            ),
                        }
                    )
            if rows:
                print_rows_table(
                    "Idea Candidates",
                    [
                        ("index", "#"),
                        ("title", "Title"),
                        ("metric", "Metric"),
                        ("citations", "Citations"),
                    ],
                    rows,
                )
        if save_json and ideas:
            self._save_json_artifact("idea.json", ideas)
        print("✅ Idea generated.")
        return ideas

    def code(
        self,
        idea: Dict[str, Any],
        baseline_results: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        if not isinstance(idea, dict) or "Experiment" not in idea:
            print("❌ Invalid idea: missing 'Experiment'. Skip coding stage.")
            return False, self.output_dir
        print("💻 Running experiments...")
        status, exp_path, error_details = self.coder.run(
            idea=idea, baseline_results=baseline_results
        )
        if status:
            print_mapping_table(
                "Coding Result",
                {
                    "Status": "success",
                    "Experiment Dir": exp_path,
                },
            )
        else:
            print_mapping_table(
                "Coding Result",
                {
                    "Status": "failed",
                    "Experiment Dir": exp_path,
                    "Error": error_details,
                },
            )
            if error_details:
                print(f"Error details: {error_details}")
        return status, exp_path

    def write(
        self,
        idea: Dict[str, Any],
        experiment_dir: str,
        save_json: bool = False,
    ) -> str:
        print("📝 Writing paper...")
        pdf_path, paper_name = self.writer.run(idea=idea, experiment_dir=experiment_dir)
        if save_json:
            self._save_json_artifact(
                "paper.json",
                {"pdf_path": pdf_path, "paper_name": paper_name},
            )
        print_mapping_table(
            "Paper Output",
            {
                "Paper Name": paper_name,
                "PDF Path": pdf_path,
            },
        )
        print("✅ Paper written.")
        return pdf_path

    def review(
        self, pdf_path: Optional[str] = None, tex_path: Optional[str] = None
    ) -> Dict[str, Any]:
        print("🔍 Reviewing paper...")
        review = self.reviewer.run(pdf_path=pdf_path, tex_path=tex_path)
        if isinstance(review, dict) and review:
            print_mapping_table("Review Summary", summarize_review(review))
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

        rows = []
        for module_name, module in modules:
            if module and hasattr(module, "cost_tracker"):
                cost = module.cost_tracker.get_total_cost()
                total_cost += cost
                rows.append({"module": module_name, "cost": f"${cost:.4f}"})

        rows.append({"module": "TOTAL", "cost": f"${total_cost:.4f}"})
        print_rows_table(
            "Pipeline Cost Summary",
            [("module", "Module"), ("cost", "Cost")],
            rows,
        )

        return total_cost

    def _save_json_artifact(self, filename: str, payload: Any) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[tiny_scientist] Saved {filename} to: {path}")
