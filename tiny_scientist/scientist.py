import json
import os
from typing import Any, Dict, Optional

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

    def think(self, intent: str, pdf_content: str = "") -> None:
        print("🧠 Generating idea...")
        # idea = self.thinker.run(intent, 1, False, pdf_content)
        idea = self.thinker.run(intent)
        self.idea = json.loads(idea)["Title"]
        idea_path = os.path.join(self.output_dir, "idea.json")
        with open(idea_path, "w") as f:
            json.dump(idea, f, indent=2)
        print("✅ Idea saved.")

    def code(self, baseline_results: Dict[str, Any]) -> None:
        print("💻 Running experiments...")
        self.baseline_results = baseline_results
        idea = {"idea": self.idea}
        self.coder.run(idea, baseline_results=baseline_results)
        print("✅ Code executed.")

    def write(self) -> None:
        print("📝 Writing paper...")
        idea = {"idea": self.idea}
        self.writer.run(idea=idea, folder_name=self.output_dir)
        print("✅ Paper written.")

    def review(self) -> None:
        print("🔍 Reviewing paper...")
        paper_name = self.idea
        pdf_name = f"{paper_name}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_name)
        self.reviewer.run(pdf_path)
        print("✅ Review complete.")
