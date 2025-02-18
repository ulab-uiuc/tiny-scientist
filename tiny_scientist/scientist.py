import os
from typing import Any, Dict, List, Optional

import yaml

from .coder import Coder
from .reviewer import Reviewer
from .thinker import Thinker
from .writer import Writer


class TinyScientist:
    def __init__(
        self,
        model: str,
        client: any,
        base_dir: str,
        name: str = "AI Scientist",
        config_path: Optional[str] = None,
        s2_api_key: Optional[str] = None
    ):
        """Initialize the AI Scientist with all its components."""
        self.name = name
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")

        # Load prompts if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.prompts = yaml.safe_load(f)

        # Initialize components
        self.thinker = Thinker(
            model=model,
            client=client,
            base_dir=base_dir,
            s2_api_key=self.s2_api_key
        )

        self.reviewer = Reviewer(
            model=model,
            client=client,
            temperature=0.75
        )

        self.writer = Writer(
            model=model,
            client=client,
            base_dir=base_dir,
            s2_api_key=self.s2_api_key
        )

    def think(
        self,
        task_description: str,
        code: str,
        skip_generation: bool = False,
        max_num_generations: int = 20,
        num_reflections: int = 5,
        check_novelty: bool = True,
        engine: str = "semanticscholar"
    ) -> List[Dict[str, Any]]:
        """Generate and evaluate research ideas."""
        # Generate initial ideas
        ideas = self.thinker.generate_ideas(
            skip_generation=skip_generation,
            max_num_generations=max_num_generations,
            num_reflections=num_reflections
        )

        # Check novelty if requested
        if check_novelty:
            ideas = self.thinker.check_idea_novelty(
                ideas=ideas,
                engine=engine
            )

        return ideas

    def think_next(
        self,
        prev_ideas: List[Dict[str, Any]] = [],
        num_reflections: int = 5,
        max_attempts: int = 10,
        check_novelty: bool = True,
        engine: str = "semanticscholar"
    ) -> List[Dict[str, Any]]:
        """Generate the next research idea based on previous ones."""
        # Generate next idea
        ideas = self.thinker.generate_next_idea(
            prev_idea_archive=prev_ideas,
            num_reflections=num_reflections,
            max_attempts=max_attempts
        )

        # Check novelty if requested
        if check_novelty:
            ideas = self.thinker.check_idea_novelty(
                ideas=ideas,
                engine=engine
            )

        return ideas

    def code(
        self,
        idea: Dict[str, Any],
        baseline_results: Dict[str, Any],
        chat_history: Optional[str] = None
    ) -> bool:
        """Implement and run experiments for an idea."""
        coder = Coder(
            base_dir=self.base_dir,
            model=self.model,
            chat_history=chat_history
        )

        return coder.perform_experiments(
            idea=idea,
            baseline_results=baseline_results
        )

    def write(
        self,
        idea: Dict[str, Any],
        folder_name: str,
        num_cite_rounds: int = 20,
        engine: str = "semanticscholar"
    ) -> None:
        """Write a research paper based on experimental results."""
        self.writer.perform_writeup(
            idea=idea,
            folder_name=folder_name,
            num_cite_rounds=num_cite_rounds,
            engine=engine
        )

    def review(
        self,
        text: str,
        num_reflections: int = 1,
        num_fs_examples: int = 1,
        num_reviews_ensemble: int = 1,
        return_msg_history: bool = False,
        reviewer_system_prompt: Optional[str] = None
    ) -> Dict:
        """Review a research paper or experimental results."""
        return self.reviewer.perform_review(
            text=text,
            num_reflections=num_reflections,
            num_fs_examples=num_fs_examples,
            num_reviews_ensemble=num_reviews_ensemble,
            return_msg_history=return_msg_history,
            reviewer_system_prompt=reviewer_system_prompt
        )

if __name__ == '__main__':
    # Example usage
    import argparse

    from .llm import AVAILABLE_LLMS, create_client

    parser = argparse.ArgumentParser(description="Run AI Scientist")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-0613",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for experiments"
    )
    args = parser.parse_args()

    # Create client and scientist
    client, model = create_client(args.model)
    scientist = Scientist(
        model=model,
        client=client,
        base_dir=args.base_dir
    )

    # Example workflow
    # 1. Generate ideas
    ideas = scientist.think(
        task_description="Improve language model training efficiency",
        code="# Your experiment code here"
    )

    # 2. Implement an idea
    if ideas:
        success = scientist.code(
            idea=ideas[0],
            baseline_results={"baseline_loss": 2.5}
        )

        # 3. Write paper if experiments successful
        if success:
            scientist.write(
                idea=ideas[0],
                folder_name=args.base_dir
            )

            # 4. Review paper
            with open(os.path.join(args.base_dir, "latex/template.tex"), "r") as f:
                paper_text = f.read()
            review = scientist.review(paper_text)
            print(review)
