from pprint import pprint
from typing import Any, Dict, Optional, Tuple

from .coder import Coder
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

    def think(self, intent: str, pdf_content: Optional[str] = None) -> Dict[str, Any]:
        pprint("🧠 Generating idea...")
        idea = self.thinker.run(intent=intent, pdf_content=pdf_content)
        pprint(idea, width=80, indent=2, compact=False)
        pprint("✅ Idea generated.")
        return idea

    def code(
        self, idea: Dict[str, Any], baseline_results: Dict[str, Any]
    ) -> Tuple[bool, str]:
        pprint("💻 Running experiments...")
        status, exp_path = self.coder.run(idea=idea, baseline_results=baseline_results)
        if status:
            pprint(f"✅ Experiment completed successfully. Results saved at {exp_path}")
        else:
            pprint(f"❌ Experiment failed. Please check {exp_path} for details.")
        return status, exp_path

    def write(self, idea: Dict[str, Any], experiment_dir: str) -> str:
        pprint("📝 Writing paper...")
        pdf_path, paper_name = self.writer.run(idea=idea, experiment_dir=experiment_dir)
        pprint(
            f"Check the generated paper named as {paper_name} and saved at {pdf_path}"
        )
        pprint("✅ Paper written.")
        return pdf_path

    def review(self, pdf_path: str) -> Dict[str, Any]:
        pprint("🔍 Reviewing paper...")
        review = self.reviewer.run(pdf_path=pdf_path)
        pprint(review, width=80, indent=2, compact=False)
        pprint("✅ Review complete.")
        return review
