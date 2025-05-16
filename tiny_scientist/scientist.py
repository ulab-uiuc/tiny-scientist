from typing import Any, Dict, List, Optional, Tuple, Union

from rich import print

from .coder import Coder
from .react_experimenter import ReactExperimenter
from .reviewer import Reviewer
from .thinker import Thinker
from .utils.input_formatter import InputFormatter
from .writer import Writer


class TinyScientist:
    def __init__(
        self,
        model: str = "gpt-4o",
        output_dir: str = "./",
        template: str = "acl",
        prompt_template_dir: Optional[str] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.template = template
        self.prompt_template_dir = prompt_template_dir
        self.input_formatter = InputFormatter()

        self.thinker = Thinker(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            tools=[],
            iter_num=3,
            search_papers=True,
            generate_exp_plan=True,
        )

        self.coder = Coder(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            max_iters=4,
            max_runs=3,
        )

        self.writer = Writer(
            model=model,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            template=template,
        )

        self.reviewer = Reviewer(
            model=model,
            prompt_template_dir=prompt_template_dir,
            tools=[],
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
        status, exp_path = self.coder.run(idea=idea, baseline_results=baseline_results)
        if status:
            print(f"✅ Experiment completed successfully. Results saved at {exp_path}")
        else:
            print(f"❌ Experiment failed. Please check {exp_path} for details.")
        return status, exp_path

    def react_experiment(
        self,
        idea: Dict[str, Any],
        domain: str = "general",
        baseline_results: Optional[Dict[str, Any]] = None,
        max_iterations: int = 10,
    ) -> Tuple[bool, str]:
        """
        Run experiments using the ReactExperimenter with domain-specific tools.
        
        Args:
            idea: Dictionary containing experiment details
            domain: Domain for the experiment (e.g., "chemistry", "physics", "general")
            baseline_results: Optional dictionary of baseline results for comparison
            max_iterations: Maximum number of ReAct iterations
            
        Returns:
            Tuple of (success, experiment_directory)
        """
        print(f"🔬 Running {domain} experiments using ReAct agent...")
        
        # Initialize ReactExperimenter with the specified domain
        reactor = ReactExperimenter(
            model=self.model,
            output_dir=self.output_dir,
            domain=domain,
            max_iterations=max_iterations,
            prompt_template_dir=self.prompt_template_dir,
        )
        
        # Run the experiment
        status, exp_path = reactor.run(idea=idea, baseline_results=baseline_results)
        
        if status:
            print(f"✅ ReAct experiment completed successfully. Results saved at {exp_path}")
        else:
            print(f"❌ ReAct experiment failed or reached max iterations. Check {exp_path} for details.")
        
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
