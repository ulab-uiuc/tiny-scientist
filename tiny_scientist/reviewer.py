import json
import os
import os.path as osp

import numpy as np
import pymupdf
import pymupdf4llm
import yaml
from pypdf import PdfReader

from .llm import (
    extract_json_between_markers,
    get_batch_responses_from_llm,
    get_response_from_llm,
)


class Reviewer:
    def __init__(self,
                 model,
                 client,
                 config_dir: str,
                 temperature=0.75):
        """Initialize the PaperReviewer with model configuration and prompt templates."""
        self.model = model
        self.client = client
        self.temperature = temperature

        # Load prompt templates
        yaml_path = osp.join(config_dir, "reviewer_prompt.yaml")
        with open(yaml_path, "r") as f:
            prompt_templates = yaml.safe_load(f)

        # Initialize prompt templates
        self.reviewer_system_prompt_base = prompt_templates.get("reviewer_system_prompt_base")
        self.reviewer_system_prompt_neg = prompt_templates.get("reviewer_system_prompt_neg")
        self.reviewer_system_prompt_pos = prompt_templates.get("reviewer_system_prompt_pos")
        template_instructions = prompt_templates.get("template_instructions")
        self.neurips_form = prompt_templates.get("neurips_form").replace(
            "{{ template_instructions }}", template_instructions
        )
        self.reviewer_reflection_prompt = prompt_templates.get("reviewer_reflection_prompt")
        self.meta_reviewer_system_prompt = prompt_templates.get("meta_reviewer_system_prompt")
        self.improvement_prompt = prompt_templates.get("improvement_prompt")

        # Initialize fewshot examples paths
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.fewshot_papers = [
            os.path.join(self.dir_path, "fewshot_examples/132_automated_relational.pdf"),
            os.path.join(self.dir_path, "fewshot_examples/attention.pdf"),
            os.path.join(self.dir_path, "fewshot_examples/2_carpe_diem.pdf"),
        ]
        self.fewshot_reviews = [
            os.path.join(self.dir_path, "fewshot_examples/132_automated_relational.json"),
            os.path.join(self.dir_path, "fewshot_examples/attention.json"),
            os.path.join(self.dir_path, "fewshot_examples/2_carpe_diem.json"),
        ]

    def perform_review(
        self,
        text,
        num_reflections=1,
        num_fs_examples=1,
        num_reviews_ensemble=1,
        msg_history=None,
        return_msg_history=False,
        reviewer_system_prompt=None,
    ):
        """Perform a review of the given text with specified parameters."""
        if reviewer_system_prompt is None:
            reviewer_system_prompt = self.reviewer_system_prompt_neg

        # Prepare base prompt
        if num_fs_examples > 0:
            fs_prompt = self._get_review_fewshot_examples(num_fs_examples)
            base_prompt = self.neurips_form + fs_prompt
        else:
            base_prompt = self.neurips_form
        base_prompt += f"\nHere is the paper you are asked to review:\n```\n{text}\n```"

        # Handle ensemble reviews
        if num_reviews_ensemble > 1:
            return self._perform_ensemble_review(
                base_prompt,
                num_reviews_ensemble,
                reviewer_system_prompt,
                msg_history,
                num_reflections,
                return_msg_history
            )

        # Handle single review
        return self._perform_single_review(
            base_prompt,
            reviewer_system_prompt,
            msg_history,
            num_reflections,
            return_msg_history
        )

    def _perform_ensemble_review(
        self,
        base_prompt,
        num_reviews_ensemble,
        reviewer_system_prompt,
        msg_history,
        num_reflections,
        return_msg_history
    ):
        """Handle ensemble review process with multiple reviewers."""
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=self.model,
            client=self.client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=self.temperature,
            n_responses=num_reviews_ensemble,
        )

        parsed_reviews = self._parse_ensemble_reviews(llm_review)
        review = self._get_meta_review(len(parsed_reviews), parsed_reviews)

        if review is None:
            review = parsed_reviews[0]

        review = self._aggregate_scores(review, parsed_reviews)

        if num_reflections > 1:
            review = self._perform_reflections(review, num_reflections, reviewer_system_prompt)

        if return_msg_history:
            msg_history = msg_histories[0][:-1]
            msg_history += [
                {
                    "role": "assistant",
                    "content": f"\nTHOUGHT:\nI will start by aggregating the opinions of {num_reviews_ensemble} reviewers that I previously obtained.\n\nREVIEW JSON:\n```json\n{json.dumps(review)}\n```"
                }
            ]
            return review, msg_history

        return review

    def _perform_single_review(
        self,
        base_prompt,
        reviewer_system_prompt,
        msg_history,
        num_reflections,
        return_msg_history
    ):
        """Handle single review process."""
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

        if num_reflections > 1:
            review = self._perform_reflections(review, num_reflections, reviewer_system_prompt)

        if return_msg_history:
            return review, msg_history

        return review

    def _parse_ensemble_reviews(self, llm_review):
        """Parse multiple reviews from ensemble review process."""
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        return [r for r in parsed_reviews if r is not None]

    def _aggregate_scores(self, review, parsed_reviews):
        """Aggregate scores from multiple reviews."""
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

        for score, limits in score_limits.items():
            scores = [
                r[score] for r in parsed_reviews
                if score in r and limits[1] >= r[score] >= limits[0]
            ]
            if scores:
                review[score] = int(round(np.mean(scores)))

        return review

    def _perform_reflections(self, review, num_reflections, reviewer_system_prompt):
        """Perform reflection iterations on the review."""
        for _ in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                self.reviewer_reflection_prompt,
                client=self.client,
                model=self.model,
                system_message=reviewer_system_prompt,
                msg_history=None,
                temperature=self.temperature,
            )
            new_review = extract_json_between_markers(text)
            assert new_review is not None, "Failed to extract JSON from LLM output"
            review = new_review
            if "I am done" in text:
                break
        return review

    def _get_meta_review(self, reviewer_count, reviews):
        """Generate meta-review from multiple reviews."""
        review_text = ""
        for i, r in enumerate(reviews):
            review_text += f"\nReview {i + 1}/{reviewer_count}:\n```\n{json.dumps(r)}\n```\n"

        base_prompt = self.neurips_form + review_text
        llm_review, _ = get_response_from_llm(
            base_prompt,
            model=self.model,
            client=self.client,
            system_message=self.meta_reviewer_system_prompt.format(reviewer_count=reviewer_count),
            print_debug=False,
            msg_history=None,
            temperature=self.temperature,
        )
        return extract_json_between_markers(llm_review)

    def _get_review_fewshot_examples(self, num_fs_examples=1):
        """Get few-shot examples for review."""
        fewshot_prompt = "\nBelow are some sample reviews, copied from previous machine learning conferences.\n"
        fewshot_prompt += "Note that while each review is formatted differently according to each reviewer's style, "
        fewshot_prompt += "the reviews are well-structured and therefore easy to navigate.\n"

        for paper, review in zip(self.fewshot_papers[:num_fs_examples], self.fewshot_reviews[:num_fs_examples]):
            txt_path = paper.replace(".pdf", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    paper_text = f.read()
            else:
                paper_text = self.load_paper(paper)
            review_text = self.load_review(review)
            fewshot_prompt += f"\nPaper:\n```\n{paper_text}\n```\n\nReview:\n```\n{review_text}\n```\n"

        return fewshot_prompt

    @staticmethod
    def load_paper(pdf_path, num_pages=None, min_size=100):
        """Load and extract text from a PDF paper."""
        try:
            if num_pages is None:
                text = pymupdf4llm.to_markdown(pdf_path)
            else:
                reader = PdfReader(pdf_path)
                min_pages = min(len(reader.pages), num_pages)
                text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
            try:
                doc = pymupdf.open(pdf_path)
                if num_pages:
                    doc = doc[:num_pages]
                text = ""
                for page in doc:
                    text += page.get_text()
                if len(text) < min_size:
                    raise Exception("Text too short")
            except Exception as e:
                print(f"Error with pymupdf, falling back to pypdf: {e}")
                reader = PdfReader(pdf_path)
                if num_pages is None:
                    text = "".join(page.extract_text() for page in reader.pages)
                else:
                    text = "".join(page.extract_text() for page in reader.pages[:num_pages])
                if len(text) < min_size:
                    raise Exception("Text too short")
        return text

    @staticmethod
    def load_review(path):
        """Load a review from a JSON file."""
        with open(path, "r") as json_file:
            loaded = json.load(json_file)
        return loaded["review"]

    def perform_improvement(self, review, coder):
        """Perform improvements based on the review using a coder."""
        formatted_prompt = self.improvement_prompt.format(review=json.dumps(review))
        return coder.run(formatted_prompt)
