import json
from typing import Any, Dict, List, Optional, Tuple

from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .tool import BaseTool, PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.input_formatter import InputFormatter
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from .utils.pricing import estimate_prompt_cost, estimate_tokens_from_text


class Reviewer:
    def __init__(
        self,
        model: str,
        tools: List[BaseTool],
        num_reviews: int = 3,
        num_reflections: int = 2,
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
        cost_tracker: Optional[BudgetChecker] = None,
        pre_reflection_threshold: float = 0.5,
        post_reflection_threshold: float = 0.8,
        s2_api_key: Optional[str] = None,
        search_papers: bool = True,
    ):
        self.tools = tools
        self.num_reviews = num_reviews
        self.num_reflections = num_reflections
        self.client, self.model = create_client(model)
        self.temperature = temperature
        self.config = Config(prompt_template_dir)
        self.search_papers = search_papers
        self.searcher: BaseTool = PaperSearchTool(
            s2_api_key=s2_api_key, engine="semanticscholar"
        )
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_related_works_string = ""
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.pre_reflection_threshold = pre_reflection_threshold
        self.post_reflection_threshold = post_reflection_threshold

        self.prompts = self.config.prompt_template.reviewer_prompt
        self.prompts.neurips_form = self.prompts.neurips_form.format(
            template_instructions=self.prompts.template_instructions
        )

    def review(self, pdf_path: Optional[str] = None, tex_path: Optional[str] = None) -> str:
        if tex_path:
            with open(tex_path, "r", encoding="utf-8") as f:
                paper_text = f.read()
            print(f"Using content from TEX file: {tex_path}")
        elif pdf_path:
            formatter = InputFormatter()
            text = formatter.parse_paper_pdf_to_json(pdf_path=pdf_path)
            paper_text = str(text)
            print(f"Using content from PDF file: {pdf_path}")
        else:
            raise ValueError("Either pdf_path or tex_path must be provided.")

        if not paper_text:
            raise ValueError("No paper text provided for review.")

        if self.search_papers:
            query = self._generate_query(paper_text)
            related_works_string = self._get_related_works(query)
            self.last_related_works_string = related_works_string
        else:
            print("[info] Skipping paper search for review (search_papers=False)")
            related_works_string = ""
            self.last_related_works_string = ""

        base_prompt = self._build_review_prompt(paper_text, related_works_string)
        system_prompt = self.prompts.reviewer_system_prompt_neg

        review, _ = self._generate_review(base_prompt, system_prompt, msg_history=[])
        self.cost_tracker.report()
        return json.dumps(review, indent=2)

    def re_review(self, review_json: str) -> str:
        current_review = json.loads(review_json)
        if not current_review:
            raise ValueError("No review provided for re-review.")

        system_prompt = self.prompts.reviewer_system_prompt_neg
        related_works_string = self.last_related_works_string

        new_review, _, _ = self._reflect_review(
            review=current_review,
            reviewer_system_prompt=system_prompt,
            related_works_string=related_works_string,
            msg_history=[],
        )
        self.cost_tracker.report()
        return json.dumps(new_review, indent=2)

    def run(self, pdf_path: Optional[str] = None, tex_path: Optional[str] = None) -> Dict[str, Any]:
        all_reviews = []

        for i in range(self.num_reviews):
            print(f"Generating {i + 1}/{self.num_reviews} review")
            current_review = self.review(pdf_path=pdf_path, tex_path=tex_path)

            # Apply tools to review
            for tool in self.tools:
                tool_input = json.dumps({"review": current_review})
                tool_output = tool.run(tool_input)
                if "review" in tool_output:
                    current_review = tool_output["review"]["review"]

            # Apply reflections with dynamic budgeting
            budget = self.cost_tracker.get_budget()
            if (
                budget is not None
                and self.cost_tracker.get_total_cost() / budget
                >= self.pre_reflection_threshold
            ):
                print("[Reviewer] Skipping review reflections due to budget limit.")
            else:
                max_rounds = self.num_reflections
                estimated_rounds = self._estimate_usable_reflection_rounds(
                    current_review
                )
                if estimated_rounds is not None:
                    if estimated_rounds <= 0:
                        print(
                            "[Reviewer] Estimated remaining budget insufficient for review reflections."
                        )
                        all_reviews.append(json.loads(current_review))
                        continue
                    max_rounds = min(max_rounds, estimated_rounds)
                rounds_done = 0
                per_round_cost = None
                while rounds_done < max_rounds:
                    start_cost = self.cost_tracker.get_total_cost()
                    current_review = self.re_review(current_review)
                    iteration_cost = self.cost_tracker.get_total_cost() - start_cost
                    if per_round_cost is None:
                        per_round_cost = max(iteration_cost, 1e-6)
                        if budget is not None:
                            allowed = budget * self.post_reflection_threshold
                            remaining = allowed - self.cost_tracker.get_total_cost()
                            additional = int(max(0.0, remaining) // per_round_cost)
                            max_rounds = min(self.num_reflections, 1 + additional)
                    rounds_done += 1
                    if (
                        budget is not None
                        and self.cost_tracker.get_total_cost()
                        >= budget * self.post_reflection_threshold
                    ):
                        break

            all_reviews.append(json.loads(current_review))

        self.cost_tracker.report()
        return self._write_meta_review(all_reviews)

    def _estimate_usable_reflection_rounds(self, current_review: str) -> Optional[int]:
        remaining_budget = self._reflection_remaining_budget()
        if remaining_budget is None:
            return None

        if remaining_budget <= 0:
            return 0

        estimated_cost = self._estimate_reflection_cost(current_review)
        if estimated_cost is None or estimated_cost <= 0:
            return None

        return int(remaining_budget // estimated_cost)

    def _reflection_remaining_budget(self) -> Optional[float]:
        candidates: List[float] = []

        module_budget = self.cost_tracker.get_budget()
        if module_budget is not None:
            allowed = module_budget * self.post_reflection_threshold
            remaining = allowed - self.cost_tracker.get_total_cost()
            candidates.append(max(0.0, remaining))

        parent_tracker = getattr(self.cost_tracker, "parent", None)
        if parent_tracker is not None:
            parent_remaining = parent_tracker.get_effective_remaining_budget()
            if parent_remaining is not None:
                candidates.append(max(0.0, parent_remaining))

        if not candidates:
            return None

        return max(0.0, min(candidates))

    def _estimate_reflection_cost(self, current_review: str) -> Optional[float]:
        try:
            prompt_preview = (
                f"Previous review: {current_review}\n"
                + self.prompts.reviewer_reflection_prompt.format(
                    related_works_string="Summaries of related work (estimated)"
                )
            )
        except Exception:
            prompt_preview = current_review

        expected_output_tokens = max(
            estimate_tokens_from_text(current_review),
            256,
        )

        return estimate_prompt_cost(
            self.model,
            [self.prompts.reviewer_system_prompt_neg, prompt_preview],
            expected_output_tokens=expected_output_tokens,
        )

    def _get_related_works(self, query: str) -> str:
        if query in self._query_cache:
            related_papers = self._query_cache[query]
        else:
            results_dict = self.searcher.run(query)
            related_papers = list(results_dict.values())
            self._query_cache[query] = related_papers if related_papers else []

        if related_papers:
            related_works_string = self._format_paper_results(related_papers)
            print("✅Related Works String Found")
        else:
            related_works_string = "No related works found."
            print("❎No Related Works Found")

        self.cost_tracker.report()
        return related_works_string

    def _build_review_prompt(self, text: str, related_works_string: str) -> str:
        base_prompt = self.prompts.neurips_form.format(
            related_works_string=related_works_string
        )
        return f"{base_prompt}\nHere is the paper you are asked to review:\n```\n{text}\n```"

    def _generate_query(self, text: str) -> str:
        query_prompt = self.prompts.query_prompt.format(paper_text=text)
        response, _ = get_response_from_llm(
            query_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.reviewer_system_prompt_neg,
            temperature=self.temperature,
            msg_history=[],
            cost_tracker=self.cost_tracker,
            task_name="generate_query",
        )
        query_data = extract_json_between_markers(response)
        self.cost_tracker.report()
        return str(query_data.get("Query", "")) if query_data else ""

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_review(
        self,
        base_prompt: str,
        reviewer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if msg_history is None:
            msg_history = []

        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=self.model,
            client=self.client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="generate_review",
        )
        review = extract_json_between_markers(llm_review)
        self.cost_tracker.report()
        return review if review is not None else {}, msg_history

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_review(
        self,
        review: Dict[str, Any],
        reviewer_system_prompt: str,
        related_works_string: str,
        msg_history: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
        updated_prompt = (
            f"Previous review: {json.dumps(review)}\n"
            + self.prompts.reviewer_reflection_prompt.format(
                related_works_string=related_works_string
            )
        )

        text, msg_history = get_response_from_llm(
            updated_prompt,
            client=self.client,
            model=self.model,
            system_message=reviewer_system_prompt,
            msg_history=msg_history,
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="reflect_review",
        )

        new_review = extract_json_between_markers(text)
        is_done = "I am done" in text

        self.cost_tracker.report()
        return new_review or {}, msg_history, is_done

    def _write_meta_review(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not reviews:
            raise ValueError("At least one review must be provided for meta-review.")

        formatted_reviews = "".join(
            f"\nReview {i + 1}:\n```\n{json.dumps(r)}\n```\n"
            for i, r in enumerate(reviews)
        )

        meta_prompt = self.prompts.neurips_form + formatted_reviews
        meta_system_prompt = self.prompts.meta_reviewer_system_prompt.format(
            reviewer_count=len(reviews)
        )

        llm_meta_review, _ = get_response_from_llm(
            meta_prompt,
            model=self.model,
            client=self.client,
            system_message=meta_system_prompt,
            msg_history=[],
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="write_meta_review",
        )

        meta_review = extract_json_between_markers(llm_meta_review)
        if meta_review is None:
            return {}

        self.cost_tracker.report()
        return self._aggregate_scores(meta_review, reviews)

    def _aggregate_scores(
        self, meta_review: Dict[str, Any], reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        score_fields = {
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

        for score, (min_val, max_val) in score_fields.items():
            valid_scores = [
                r[score]
                for r in reviews
                if score in r
                and isinstance(r[score], (int, float))
                and min_val <= r[score] <= max_val
            ]

            if valid_scores:
                meta_review[score] = int(round(sum(valid_scores) / len(valid_scores)))

        self.cost_tracker.report()
        return meta_review

    @staticmethod
    def _format_paper_results(papers: List[Dict[str, Any]]) -> str:
        """Format paper results with title, authors, venue, and abstract"""
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            title = paper.get("title", "No title")
            source = paper.get("source", "No authors")
            info = paper.get("info", "No venue")
            abstract = paper.get("abstract", "")

            # Format: Title. Authors. Venue.\nAbstract: ...
            paper_str = f"{i}: {title}. {source}. {info}"
            if abstract and len(abstract.strip()) > 0:
                # Truncate very long abstracts
                abstract_text = abstract.strip()
                if len(abstract_text) > 500:
                    abstract_text = abstract_text[:500] + "..."
                paper_str += f"\nAbstract: {abstract_text}"

            paper_strings.append(paper_str)

        return "\n\n".join(paper_strings)
