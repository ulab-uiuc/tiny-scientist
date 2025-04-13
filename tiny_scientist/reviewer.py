import json
from typing import Any, Dict, List, Optional, Tuple

from .configs import Config
from .tool import BaseTool, PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.input_formatter import InputFormatter
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)


class Reviewer:
    def __init__(
        self,
        model: str,
        tools: List[BaseTool],
        num_reviews: int = 3,  # Number of separate reviews to generate
        num_reflections: int = 2,  # Number of re_review calls per review
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
    ):
        self.tools = tools
        self.num_reviews = num_reviews
        self.num_reflections = num_reflections
        self.client, self.model = create_client(model)
        self.temperature = temperature
        # Initialize the searcher and set s2_api_key
        self.config = Config(prompt_template_dir)
        self.searcher = PaperSearchTool()
        # Cache for queries to avoid duplicate searches
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_related_works_string = ""
        # Load prompt templates from configuration file

        self.prompts = self.config.prompt_template.reviewer_prompt
        self.prompts.neurips_form = self.prompts.neurips_form.format(
            template_instructions=self.prompts.template_instructions
        )

    def review(self, pdf_path: str) -> str:
        """
        Generate an initial review for the given paper.
        Expects a "text" field in the intent (the paper content).
        Generates a query from the paper text, retrieves related works using run(),
        and incorporates that context into the review prompt.
        """
        formatter = InputFormatter()
        text = formatter.parse_paper_pdf_to_json(pdf_path=pdf_path)
        print(f"Using content from PDF file: {pdf_path}")

        if not text:
            raise ValueError("No paper text provided for review.")

        # Generate a search query based on the paper text.
        query = self._generate_query(str(text))
        print(f"Generated Query: {query}")

        # Retrieve related papers using run() and cache the results.
        if query in self._query_cache:
            related_papers = self._query_cache[query]
        else:
            results_dict = self.searcher.run(query)  # run() returns a dict
            related_papers = list(results_dict.values())
            self._query_cache[query] = related_papers if related_papers else []

        # Format search results using the static method.
        if related_papers:
            related_works_string = self._format_paper_results(related_papers)
            print("✅Related Works String Found")
        else:
            related_works_string = "No related works found."
            print("❎No Related Works Found")

        self.last_related_works_string = related_works_string

        # Construct the base prompt by inserting the related works.
        base_prompt = self.prompts.neurips_form.format(
            related_works_string=related_works_string
        )
        base_prompt += (
            "\nHere is the paper you are asked to review:\n```\n" + str(text) + "\n```"
        )

        system_prompt = self.prompts.reviewer_system_prompt_neg
        review, msg_history = self._generate_review(
            base_prompt, system_prompt, msg_history=[]
        )
        return json.dumps(review, indent=2)

    def re_review(self, review_json: str) -> str:
        """
        Refine an existing review using a reflection prompt.
        Expects the info dictionary to contain a "review" field.
        """
        current_review = json.loads(review_json)
        if not current_review:
            raise ValueError("No review provided for re-review.")
        system_prompt = self.prompts.reviewer_system_prompt_neg
        # Use the current review to extract the query if available.
        related_works_string = getattr(self, "last_related_works_string", "")
        new_review, msg_history, is_done = self._reflect_review(
            review=current_review,
            reviewer_system_prompt=system_prompt,
            related_works_string=related_works_string,
            msg_history=[],
        )
        return json.dumps(new_review, indent=2)

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """
        Execute the review process in an ensemble fashion.
        For each of num_reviews iterations, generate a review using review() and then refine it by calling
        re_review() num_reflections times. Finally, aggregate all generated reviews into a meta-review.
        """
        all_reviews = []
        for i in range(self.num_reviews):
            print(f"Generating {i + 1}/{self.num_reviews} review")
            current_review = self.review(pdf_path)
            for tool in self.tools:
                tool_input = json.dumps({"review": current_review})
                tool_output = tool.run(tool_input)
                # Expect tool_output to contain a "review" key.
                if "review" in tool_output:
                    current_review = tool_output["review"]["review"]
            for _ in range(self.num_reflections):
                current_review = self.re_review(current_review)
            all_reviews.append(json.loads(current_review))
        final_meta_review = self._write_meta_review(all_reviews)
        return final_meta_review

    def _generate_query(self, text: str) -> str:
        """
        Generate a concise search query based on the paper text.
        """
        query_prompt = self.prompts.query_prompt.format(paper_text=text)
        response, _ = get_response_from_llm(
            query_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.reviewer_system_prompt_neg,
            temperature=self.temperature,
            msg_history=[],
        )
        query_data = extract_json_between_markers(response)
        if query_data is None or "Query" not in query_data:
            return ""
        return str(query_data["Query"])

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_review(
        self,
        base_prompt: str,
        reviewer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a review from the provided prompt.
        This function generates a review in a single step.
        """
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
        return review if review is not None else {}, msg_history

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_review(
        self,
        review: Dict[str, Any],
        reviewer_system_prompt: str,
        related_works_string: str,
        msg_history: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """
        Refine the given review using a reflection prompt.
        The current review is included in the prompt to provide context.
        """

        # Prepend the current review context.
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
        )
        new_review = extract_json_between_markers(text)
        is_done = "I am done" in text
        return new_review, msg_history, is_done

    def _write_meta_review(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a meta-review by aggregating multiple individual reviews.
        This function takes a list of review dictionaries, formats them,
        and uses the LLM to produce an aggregated meta-review.
        It then aggregates numerical scores via _aggregate_scores.
        """
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
        )
        meta_review = extract_json_between_markers(llm_meta_review)
        if meta_review is None:
            return {}
        result = self._aggregate_scores(meta_review, reviews)
        return result

    def _aggregate_scores(
        self, meta_review: Dict[str, Any], reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate numerical scores from multiple reviews.
        For each score field, compute the mean (rounded to the nearest integer).
        Expected score fields include: Originality, Quality, Clarity, Significance,
        Soundness, Presentation, Contribution, Overall, and Confidence.
        """
        score_limits: Dict[str, Tuple[int, int]] = {
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
                r[score]
                for r in reviews
                if score in r
                and isinstance(r[score], (int, float))
                and min_val <= r[score] <= max_val
            ]
            if valid_scores:
                meta_review[score] = int(round(sum(valid_scores) / len(valid_scores)))
        return meta_review

    @staticmethod
    def _format_paper_results(papers: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format a list of paper dictionaries into a human-readable string.
        Each paper is represented by its index, title, authors, venue, year, and abstract.
        """
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"{i}: {paper.get('title', 'No title')}. {paper.get('source', 'No authors')}. "
                f"{paper.get('info', 'No venue')}"
            )
        return "\n\n".join(paper_strings)
