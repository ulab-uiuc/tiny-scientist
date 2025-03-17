import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .llm import extract_json_between_markers, get_response_from_llm
from .searcher import PaperSearcher
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.loader import input_formatter


class Reviewer:
    def __init__(self, model: Any, client: Any, config_dir: str, temperature: float = 0.75):
        """Initialize the Reviewer with model configuration and prompt templates."""
        self.model = model
        self.client = client
        self.temperature = temperature
        self.searcher = PaperSearcher()
        self.input_formatter = input_formatter()
        # Load prompt templates
        with open(osp.join(config_dir, "reviewer_prompt.yaml"), "r") as f:
            self.prompts = yaml.safe_load(f)

        # Process template instructions in neurips form
        if "template_instructions" in self.prompts and "neurips_form" in self.prompts:
            self.prompts["neurips_form"] = self.prompts["neurips_form"].replace(
                "{{ template_instructions }}", self.prompts["template_instructions"]
            )

        # Set example paths
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self._set_fewshot_paths()

    def _set_fewshot_paths(self):
        """Initialize paths for few-shot examples."""
        base = os.path.join(self.dir_path, "fewshot_examples")
        self.fewshot_papers = [
            os.path.join(base, "132_automated_relational.pdf"),
            os.path.join(base, "attention.pdf"),
            os.path.join(base, "2_carpe_diem.pdf"),
        ]
        self.fewshot_reviews = [
            os.path.join(base, "132_automated_relational.json"),
            os.path.join(base, "attention.json"),
            os.path.join(base, "2_carpe_diem.json"),
        ]

    def write_review(
            self,
            text: str,
            num_reflections: int = 2,
            num_fs_examples: int = 1,
            msg_history: Optional[List[Dict]] = None,
            return_msg_history: bool = False,
            reviewer_system_prompt: Optional[str] = None,
    ) -> Any:
        """Write a review for the given text with specified parameters."""
        # Use default system prompt if none provided
        if reviewer_system_prompt is None:
            reviewer_system_prompt = self.prompts.get("reviewer_system_prompt_neg")

        # Prepare base prompt with optional few-shot examples
        base_prompt = self._prepare_base_prompt(text, num_fs_examples, "")

        # Generate review
        review, updated_msg_history = self._generate_review(
            base_prompt, reviewer_system_prompt,
            msg_history, num_reflections
        )

        return (review, updated_msg_history) if return_msg_history else review

    def write_meta_review(
            self,
            reviews: List[Dict],
            reviewer_system_prompt: Optional[str] = None
    ) -> Dict:
        if not reviews:
            raise ValueError("At least one review must be provided")

        # Use default meta-reviewer system prompt if none provided
        if reviewer_system_prompt is None:
            reviewer_system_prompt = self.prompts.get("meta_reviewer_system_prompt").format(
                reviewer_count=len(reviews)
            )

        # Format all reviews for the meta-reviewer
        review_text = "".join(
            f"\nReview {i + 1}/{len(reviews)}:\n```\n{json.dumps(r)}\n```\n"
            for i, r in enumerate(reviews)
        )

        # Get meta-review from LLM
        llm_review, _ = get_response_from_llm(
            self.prompts["neurips_form"] + review_text,
            model=self.model,
            client=self.client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=None,
            temperature=self.temperature,
        )

        meta_review = extract_json_between_markers(llm_review)
        return self._aggregate_scores(meta_review, reviews)

    def _prepare_base_prompt(self, text: str, num_fs_examples: int, related_works_string: str = "") -> str:
        """Prepare the base prompt with optional few-shot examples."""
        if num_fs_examples > 0:
            fs_prompt = self._get_review_fewshot_examples(num_fs_examples)
            base_prompt = self.prompts["neurips_form"].format(related_works_string=related_works_string) + fs_prompt
        else:
            base_prompt = self.prompts["neurips_form"].format(related_works_string=related_works_string)

        return base_prompt + f"\nHere is the paper you are asked to review:\n```\n{text}\n```"

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_review(
            self,
            base_prompt: str,
            reviewer_system_prompt: str,
            msg_history: Optional[List[Dict]],
            num_reflections: int
    ) -> tuple[Dict, List[Dict]]:
        """Generate a review with optional reflections."""
        # Generate initial review
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=self.model,
            client=self.client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=self.temperature,
        )
        review = extract_json_between_markers(llm_review)
        query = review.get("Query", "")

        if num_reflections > 1:
            for j in range(num_reflections - 1):
                current_round = j + 2
                new_review, msg_history, is_done = self._reflect_review(
                    review=review, current_round=current_round, num_reflections=num_reflections,
                    reviewer_system_prompt=reviewer_system_prompt,
                    query=query, msg_history=msg_history)

                if not new_review:
                    break

                review = new_review

                if is_done:
                    break

        return review, msg_history

    def _aggregate_scores(self, meta_review: Dict, reviews: List[Dict]) -> Dict:
        """Aggregate numerical scores from multiple reviews."""
        score_limits = {
            "Originality": (1, 4),
            "Quality": (1, 4),
            "Clarity": (1, 4),
            "Significance": (1, 4),
            "Soundness": (1, 4),
            "Presentation": (1, 4),
            "Contribution": (1, 4),
            "Overall": (1, 10),
            "Confidence": (1, 5),
        }

        for score, (min_val, max_val) in score_limits.items():
            valid_scores = [
                r[score] for r in reviews
                if score in r and min_val <= r[score] <= max_val
            ]
            if valid_scores:
                meta_review[score] = int(round(np.mean(valid_scores)))

        return meta_review

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_review(self, review: Dict, current_round: int, num_reflections: int,
                        reviewer_system_prompt: str, query: str,
                        msg_history: List[Dict]) -> Tuple[Optional[Dict], List[Dict], bool]:
        """Perform a reflection iteration to improve the review using related works."""

        related_works_string = self.searcher.search_for_papers(query, result_limit=5)

        text, msg_history = get_response_from_llm(
            self.prompts["reviewer_reflection_prompt"].format(
                current_round=current_round,
                num_reflections=num_reflections,
                related_works_string=related_works_string
            ),
            client=self.client,
            model=self.model,
            system_message=reviewer_system_prompt,
            msg_history=msg_history,
            temperature=self.temperature,
        )

        new_review = extract_json_between_markers(text)

        is_done = "I am done" in text

        return new_review, msg_history, is_done

    def _get_review_fewshot_examples(self, num_fs_examples: int = 1) -> str:
        """Get few-shot examples formatted for the prompt."""
        fewshot_prompt = "\nBelow are some sample reviews, copied from previous machine learning conferences.\n"
        fewshot_prompt += "Note that while each review is formatted differently according to each reviewer's style, "
        fewshot_prompt += "the reviews are well-structured and therefore easy to navigate.\n"

        # Include requested number of examples
        for paper_path, review_path in zip(
                self.fewshot_papers[:num_fs_examples],
                self.fewshot_reviews[:num_fs_examples]
        ):
            # Try to load pre-extracted text first
            txt_path = paper_path.replace(".pdf", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    paper_text = f.read()
            else:
                paper_text = self.input_formatter.parse_paper_pdf_to_json(paper_path)

            review_text = self.input_formatter.parse_review_json(review_path)
            fewshot_prompt += f"\nPaper:\n```\n{paper_text}\n```\n\nReview:\n```\n{review_text}\n```\n"

        return fewshot_prompt
