from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio

from rich import print

from .coder import Coder
from .reviewer import Reviewer
from .thinker import Thinker
from .utils.cost_tracker import CostTracker
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
        use_mcp: bool = True,
    ):
        self.model = model
        self.output_dir = output_dir
        self.template = template
        self.prompt_template_dir = prompt_template_dir
        self.input_formatter = InputFormatter()
        self.use_mcp = use_mcp

        self.cost = 0.0

        # Initialize MCP client if enabled
        self.mcp_client = MCPClient() if use_mcp else None

        # Naive budget split
        modules = ["thinker", "coder", "writer", "reviewer"]
        per_module_budget = budget / len(modules) if budget else None

        self.thinker = Thinker(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            iter_num=3,
            search_papers=True,
            generate_exp_plan=True,
            enable_ethical_defense=False,
            cost_tracker=CostTracker(budget=per_module_budget),
            mcp_client=self.mcp_client,
        )

        self.coder = Coder(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            max_iters=4,
            max_runs=3,
            cost_tracker=CostTracker(budget=per_module_budget),
            mcp_client=self.mcp_client,
        )

        self.writer = Writer(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            template=template,
            cost_tracker=CostTracker(budget=per_module_budget),
            mcp_client=self.mcp_client,
        )

        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            cost_tracker=CostTracker(budget=per_module_budget),
            mcp_client=self.mcp_client,
        )

    async def initialize_mcp(self):
        """Initialize MCP servers."""
        if self.mcp_client:
            print("🔧 Initializing MCP servers...")
            results = await self.mcp_client.start_all_servers()
            for server_name, success in results.items():
                if success:
                    print(f"✅ MCP server '{server_name}' started successfully")
                else:
                    print(f"❌ Failed to start MCP server '{server_name}'")

    async def cleanup_mcp(self):
        """Clean up MCP servers."""
        if self.mcp_client:
            print("🧹 Shutting down MCP servers...")
            await self.mcp_client.stop_all_servers()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup_mcp()

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
