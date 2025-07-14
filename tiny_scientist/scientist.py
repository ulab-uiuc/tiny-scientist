import datetime
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
from .utils.mcp_client import MCPClient
from .writer import Writer


class TinyScientist:
    def __init__(
        self,
        model: str = "gpt-4o",
        output_dir: str = "./",
        template: str = "acl",
        prompt_template_dir: Optional[str] = None,
        budget: Optional[float] = None,
        enable_safety_check: bool = True,
        budget_preference: Optional[str] = None,
        use_mcp: bool = True,
    ):
        self.model = model
        self.base_output_dir = output_dir  # Store user's base directory

        # Create a unique experiment directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")

        # Ensure the experiment directory exists
        os.makedirs(self.experiment_dir, exist_ok=True)
        print(f"🔬 Created experiment directory: {self.experiment_dir}")

        self.template = template
        self.prompt_template_dir = prompt_template_dir
        self.input_formatter = InputFormatter()
        self.enable_safety_check = enable_safety_check
        self.use_mcp = use_mcp

        self.cost = 0.0

        # Initialize MCP client if enabled
        self.mcp_client = MCPClient() if use_mcp else None

        # Naive budget split
        modules = ["safety_checker", "thinker", "coder", "writer", "reviewer"]
        per_module_budget = budget / len(modules) if budget else None
        if budget_preference is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config.toml"
            )
            if os.path.exists(config_path):
                cfg = toml.load(config_path)
                budget_preference = cfg.get("core", {}).get(
                    "budget_preference", "balanced"
                )
            else:
                budget_preference = "balanced"

        weights = {
            "balanced": {"thinker": 0.3, "writer": 0.3, "reviewer": 0.3, "coder": 0.1},
            "write-heavy": {
                "thinker": 0.2,
                "writer": 0.5,
                "reviewer": 0.2,
                "coder": 0.1,
            },
            "think-heavy": {
                "thinker": 0.5,
                "writer": 0.2,
                "reviewer": 0.2,
                "coder": 0.1,
            },
            "review-heavy": {
                "thinker": 0.2,
                "writer": 0.2,
                "reviewer": 0.5,
                "coder": 0.1,
            },
        }
        if budget_preference not in weights:
            raise ValueError(f"Unknown budget preference: {budget_preference}")

        allocation = {
            k: (budget * w if budget is not None else None)
            for k, w in weights[budget_preference].items()
        }

        self.safety_checker = (
            SafetyChecker(
                model=model, cost_tracker=BudgetChecker(budget=per_module_budget)
            )
            if enable_safety_check
            else None
        )

        # Use the unique experiment directory for all modules
        self.thinker = Thinker(
            model=model,
            output_dir=self.experiment_dir,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            iter_num=3,
            search_papers=True,
            generate_exp_plan=True,
            enable_ethical_defense=False,
            enable_safety_check=enable_safety_check,
            cost_tracker=BudgetChecker(budget=allocation.get("thinker")),
            mcp_client=self.mcp_client,
        )

        self.coder = Coder(
            model=model,
            output_dir=self.experiment_dir,
            prompt_template_dir=prompt_template_dir,
            max_iters=4,
            max_runs=3,
            cost_tracker=BudgetChecker(budget=allocation.get("coder")),
            mcp_client=self.mcp_client,
        )

        self.writer = Writer(
            model=model,
            output_dir=self.experiment_dir,
            prompt_template_dir=prompt_template_dir,
            template=template,
            cost_tracker=BudgetChecker(budget=allocation.get("writer")),
            mcp_client=self.mcp_client,
        )

        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            cost_tracker=BudgetChecker(budget=allocation.get("reviewer")),
            mcp_client=self.mcp_client,
        )

    async def initialize_mcp(self) -> None:
        """Initialize MCP servers."""
        if self.mcp_client:
            print("🔧 Initializing MCP servers...")
            results = await self.mcp_client.start_all_servers()
            for server_name, success in results.items():
                if success:
                    print(f"✅ MCP server '{server_name}' started successfully")
                else:
                    print(f"❌ Failed to start MCP server '{server_name}'")

    async def cleanup_mcp(self) -> None:
        """Clean up MCP servers."""
        if self.mcp_client:
            print("🧹 Shutting down MCP servers...")
            await self.mcp_client.stop_all_servers()

    async def __aenter__(self) -> "TinyScientist":
        """Async context manager entry."""
        await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup_mcp()

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
        return status, self.experiment_dir

    def write(self, idea: Dict[str, Any], experiment_dir: Optional[str] = None) -> str:
        print("📝 Writing paper...")
        # Use the internal experiment directory if no specific directory is provided
        exp_dir = experiment_dir if experiment_dir is not None else self.experiment_dir
        pdf_path, paper_name = self.writer.run(idea=idea, experiment_dir=exp_dir)
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
