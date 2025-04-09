import json
import os
from typing import Any, Dict

from tiny_scientist.coder import Coder
from tiny_scientist.reviewer import Reviewer
from tiny_scientist.thinker import Thinker
from tiny_scientist.utils.loader import load_paper
from tiny_scientist.writer import Writer


class TinyScientist:
    def __init__(self, model: str, client: Any, base_dir: str, template: str = "acl"):
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.template = template

        self.thinker = Thinker(
            tools=[],
            iter_num=3,
            model=model,
            client=client,
            base_dir=base_dir,
            s2_api_key=os.getenv("S2_API_KEY")
        )

        self.coder = Coder(
            base_dir=base_dir,
            model=model,
            max_iters=4,
            max_runs=3
        )

        self.writer = Writer(
            model=model,
            client=client,
            base_dir=base_dir,
            template=template,
        )

        self.reviewer = Reviewer(
            model=model,
            client=client,
            tools=[],
        )

    def think(self, intent: Dict[str, Any]) -> None:
        print("🧠 Generating idea...")
        ideas = self.thinker.think(intent, check_novelty=False, pdf_content="")
        self.idea = ideas[0] if isinstance(ideas, list) else ideas
        idea_path = os.path.join(self.base_dir, "idea.json")
        with open(idea_path, "w") as f:
            json.dump(self.idea, f, indent=2)
        print("✅ Idea saved.")

    def code(self, baseline_results: Dict[str, Any]) -> None:
        print("💻 Running experiments...")
        self.baseline_results = baseline_results
        idea = self.idea.get("idea", self.idea)[0]
        self.coder.perform_experiments(idea, baseline_results=baseline_results)
        print("✅ Code executed.")

    def write(self) -> None:
        print("📝 Writing paper...")
        idea = self.idea.get("idea", self.idea)[0]
        self.writer.run(idea=idea, folder_name=self.base_dir)
        print("✅ Paper written.")

    def review(self) -> None:
        print("🔍 Reviewing paper...")
        paper_name = self.idea.get("idea", self.idea)[0]["Title"]
        pdf_name = f"{paper_name}.pdf"
        pdf_path = os.path.join(self.base_dir, pdf_name)
        text = load_paper(pdf_path)
        self.reviewer.run({"text": text})
        print("✅ Review complete.")
