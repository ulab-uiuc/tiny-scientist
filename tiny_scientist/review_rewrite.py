import json
from typing import Any, Dict, List, Optional, Tuple

from rich import print

from .configs import Config
from .tool import BaseTool, PaperSearchTool # Assuming PaperSearchTool might still be used for context
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.input_formatter import InputFormatter
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)

class ReviewRewriter: # Renamed class
    def __init__(
        self,
        model: str,
        tools: List[BaseTool], # tools might be used by original review part or by ethical review
        num_reviews: int = 1, # Defaulting to 1 initial review before ethical assessment
        num_reflections: int = 1, # Defaulting to 1 reflection for the initial review
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
    ):
        self.tools = tools
        self.num_reviews = num_reviews
        self.num_reflections = num_reflections
        self.client, self.model = create_client(model)
        self.temperature = temperature
        self.config = Config(prompt_template_dir)
        self.searcher = PaperSearchTool() # Retaining searcher for potential contextual use
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_related_works_string = "" # For context in initial review stages

        self.prompts = self.config.prompt_template.reviewer_prompt
        # Ensure neurips_form is formatted; specific ethical review forms will be new
        if hasattr(self.prompts, 'neurips_form') and hasattr(self.prompts, 'template_instructions'):
            self.prompts.neurips_form = self.prompts.neurips_form.format(
                template_instructions=self.prompts.template_instructions
            )
        else:
            print("[WARNING] NeurIPS form or template instructions not found in prompts. Standard review might be affected.")

    def _perform_ethical_review(self, paper_text: str) -> Dict[str, Any]:
        """Performs a detailed ethical review of the paper.
        This will use specific prompts to guide the LLM through various ethical dimensions.
        """
        print("[INFO] Performing detailed ethical review...")
        # This prompt will be defined in reviewer_prompt.yaml
        ethical_review_prompt_template = self.prompts.get('ethical_review_guidelines_prompt', '')
        if not ethical_review_prompt_template:
            print("[ERROR] Ethical review guidelines prompt not found!")
            return {"error": "Ethical review guidelines prompt missing."}

        ethical_review_prompt = ethical_review_prompt_template.format(paper_text=paper_text)
        
        system_prompt = self.prompts.get('ethical_reviewer_system_prompt', self.prompts.reviewer_system_prompt_base)

        response, _ = get_response_from_llm(
            msg=ethical_review_prompt,
            client=self.client,
            model=self.model,
            system_message=system_prompt,
            temperature=self.temperature, # May use a different temperature for ethical review
        )
        
        ethical_review_data = extract_json_between_markers(response)
        if ethical_review_data:
            print("[INFO] Ethical review data extracted.")
            return ethical_review_data
        else:
            print("[WARNING] Could not extract structured ethical review data. Returning raw response.")
            return {"raw_ethical_review": response}

    def rewrite_paper_based_on_ethical_feedback(self, paper_text: str, ethical_review_feedback: Dict[str, Any]) -> str:
        """Rewrites the paper content based on ethical review feedback.
        This will use specific prompts to guide the LLM.
        """
        print("[INFO] Rewriting paper based on ethical feedback...")
        rewrite_prompt_template = self.prompts.get('rewrite_paper_instruction_prompt', '')
        if not rewrite_prompt_template:
            print("[ERROR] Rewrite paper instruction prompt not found!")
            return paper_text # Return original if no prompt

        # Convert dict feedback to string, perhaps JSON string or a formatted summary
        feedback_str = json.dumps(ethical_review_feedback, indent=2)

        rewrite_prompt = rewrite_prompt_template.format(paper_text=paper_text, ethical_feedback=feedback_str)
        system_prompt = self.prompts.get('rewrite_paper_system_prompt', self.prompts.reviewer_system_prompt_base) # Placeholder

        rewritten_text, _ = get_response_from_llm(
            msg=rewrite_prompt,
            client=self.client,
            model=self.model,
            system_message=system_prompt,
            temperature=self.temperature, # May use a different temperature for rewriting
        )
        # Assuming rewritten_text is the full rewritten paper. Post-processing might be needed.
        if rewritten_text:
            print("[INFO] Paper rewritten based on ethical feedback.")
            return rewritten_text
        else:
            print("[WARNING] Paper rewriting failed or returned empty. Returning original text.")
            return paper_text

    def _write_final_meta_review(self, paper_text: str, initial_reviews: List[Dict[str, Any]], ethical_review: Dict[str, Any], rewritten_paper_text: Optional[str] = None) -> Dict[str, Any]:
        """Writes a final meta-review including academic and ethical assessments."""
        print("[INFO] Writing final meta-review...")
        if not initial_reviews and not ethical_review:
            raise ValueError("At least initial reviews or an ethical review must be provided for meta-review.")

        formatted_initial_reviews = "".join(
            f"\nInitial Review {i + 1}:\n```json\n{json.dumps(r)}\n```\n"
            for i, r in enumerate(initial_reviews)
        )
        
        formatted_ethical_review = f"\nEthical Review:\n```json\n{json.dumps(ethical_review)}\n```\n"

        meta_review_prompt_template = self.prompts.get('final_meta_review_prompt', self.prompts.neurips_form) 
        # This prompt template needs to be designed to accept paper_text, initial_reviews, ethical_review, and optionally rewritten_paper_text
        # and guide the LLM to produce a JSON with academic scores and ethical scores.

        # For the meta-review, we might want to provide the most relevant version of the paper text
        final_paper_text_for_meta_review = rewritten_paper_text if rewritten_paper_text else paper_text

        meta_prompt = meta_review_prompt_template.format(
            paper_text=final_paper_text_for_meta_review,
            initial_reviews_summary=formatted_initial_reviews,
            ethical_review_summary=formatted_ethical_review,
            # The prompt should define how to use these sections
            # It should also include template_instructions for the JSON output format
            # which now needs to include ethical scores.
        )

        meta_system_prompt = self.prompts.get('meta_reviewer_system_prompt', self.prompts.reviewer_system_prompt_base)
        # Adjust system prompt if it needs to be aware of the ethical + academic scope
        meta_system_prompt = meta_system_prompt.format(reviewer_count=len(initial_reviews) if initial_reviews else 1) 

        llm_meta_review, _ = get_response_from_llm(
            meta_prompt,
            model=self.model,
            client=self.client,
            system_message=meta_system_prompt,
            msg_history=[],
            temperature=self.temperature,
        )

        final_meta_review_data = extract_json_between_markers(llm_meta_review)
        if final_meta_review_data is None:
            print("[WARNING] Could not extract structured final meta-review. Returning raw response.")
            return {"raw_final_meta_review": llm_meta_review}

        # Aggregation of scores will need to be adapted if the structure from LLM changes.
        # This is a placeholder for now, assuming academic scores come from initial_reviews
        # and ethical scores are part of final_meta_review_data directly from LLM.
        # Or, _aggregate_scores could be enhanced.
        if initial_reviews: # Try to aggregate academic scores if initial reviews exist
             final_meta_review_data = self._aggregate_scores(final_meta_review_data, initial_reviews) 

        print("[INFO] Final meta-review generated.")
        return final_meta_review_data

    # --- Methods from original Reviewer class --- 
    # These methods might be reused or adapted for the initial review part of the process.
    # The `review` and `re_review` methods would constitute the "initial academic review".

    def _initial_academic_review(self, pdf_path: str) -> str: # Renamed from 'review'
        formatter = InputFormatter()
        text = formatter.parse_paper_pdf_to_json(pdf_path=pdf_path)
        paper_text = str(text)
        print(f"[INFO] Generating initial academic review for content from PDF file: {pdf_path}")

        if not paper_text:
            raise ValueError("No paper text provided for initial academic review.")

        query = self._generate_query(paper_text) # For related works
        related_works_string = self._get_related_works(query)
        self.last_related_works_string = related_works_string # Save for potential re-review

        base_prompt = self._build_review_prompt(paper_text, related_works_string) # Standard review prompt
        system_prompt = self.prompts.get('reviewer_system_prompt_neg', self.prompts.reviewer_system_prompt_base)

        review_data, _ = self._generate_review(base_prompt, system_prompt, msg_history=[])
        return json.dumps(review_data, indent=2)

    def _reflect_initial_review(self, review_json: str) -> str: # Renamed from 're_review'
        current_review = json.loads(review_json)
        if not current_review:
            raise ValueError("No initial review provided for reflection.")

        system_prompt = self.prompts.get('reviewer_system_prompt_neg', self.prompts.reviewer_system_prompt_base)
        related_works_string = self.last_related_works_string # Use related works from initial review

        new_review, _, _ = self._reflect_review_academic(
            review=current_review,
            reviewer_system_prompt=system_prompt,
            related_works_string=related_works_string,
            msg_history=[],
        )
        return json.dumps(new_review, indent=2)

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """Main orchestration method."""
        formatter = InputFormatter()
        original_paper_text_dict = formatter.parse_paper_pdf_to_json(pdf_path=pdf_path)
        original_paper_text = str(original_paper_text_dict)

        if not original_paper_text:
            print("[ERROR] Failed to parse paper PDF or PDF is empty.")
            return {"error": "Failed to parse paper PDF."}

        print(f"[INFO] Starting Review-Rewrite process for: {pdf_path}")

        # 1. Initial Academic Review(s)
        all_initial_reviews_json = []
        current_academic_review_str = "{}" # Initialize as empty JSON string
        for i in range(self.num_reviews):
            print(f"[INFO] Generating initial academic review {i + 1}/{self.num_reviews}")
            current_academic_review_str = self._initial_academic_review(pdf_path) # pdf_path is still used here
            
            # Optional: Apply tools to the academic review (if any tools are configured for this)
            # This part is kept from original logic, assuming tools operate on the review JSON string.
            temp_review_dict = json.loads(current_academic_review_str)
            for tool in self.tools:
                # Assuming tool.run expects a string dump of a dict with a "review" key
                # This might need adjustment based on actual tool interface
                tool_input_dict = {"review": temp_review_dict} 
                tool_output = tool.run(json.dumps(tool_input_dict)) # Pass JSON string
                if isinstance(tool_output, dict) and "review" in tool_output and "review" in tool_output["review"]:
                    temp_review_dict = tool_output["review"]["review"] # Assuming nested structure
                elif isinstance(tool_output, str): # If tool directly returns the modified review JSON string
                    try: 
                        temp_review_dict = json.loads(tool_output) 
                    except json.JSONDecodeError:
                        print(f"[WARNING] Tool {tool.name if hasattr(tool, 'name') else 'Unknown Tool'} returned non-JSON string.")
            current_academic_review_str = json.dumps(temp_review_dict)

            # Apply reflections to the academic review
            for j in range(self.num_reflections):
                print(f"[INFO] Reflecting on initial academic review {i + 1} (Reflection {j + 1}/{self.num_reflections})")
                current_academic_review_str = self._reflect_initial_review(current_academic_review_str)
            
            all_initial_reviews_json.append(json.loads(current_academic_review_str))

        # 2. Detailed Ethical Review
        ethical_review_output = self._perform_ethical_review(original_paper_text)

        # 3. Rewrite Paper based on Ethical Feedback
        rewritten_paper_text_str = self.rewrite_paper_based_on_ethical_feedback(original_paper_text, ethical_review_output)

        # 4. (Optional) Brief assessment of rewritten paper - can be added later if needed

        # 5. Final Meta-Review (including ethical assessment)
        # Pass original_paper_text and rewritten_paper_text_str for the meta-review to consider both.
        final_meta_report = self._write_final_meta_review(
            paper_text=original_paper_text, # Provide original for context
            initial_reviews=all_initial_reviews_json,
            ethical_review=ethical_review_output,
            rewritten_paper_text=rewritten_paper_text_str
        )

        print("[INFO] Review-Rewrite process completed.")
        return {
            "initial_academic_reviews": all_initial_reviews_json,
            "ethical_review": ethical_review_output,
            "rewritten_paper_content": rewritten_paper_text_str, # Storing the rewritten text
            "final_meta_review": final_meta_report
        }

    # --- Helper methods from original Reviewer class (potentially adapted) ---
    def _get_related_works(self, query: str) -> str:
        if query in self._query_cache:
            related_papers = self._query_cache[query]
        else:
            results_dict = self.searcher.run(query)
            related_papers = list(results_dict.values()) if results_dict else [] # Ensure it's a list
            self._query_cache[query] = related_papers

        if related_papers:
            related_works_string = self._format_paper_results(related_papers)
            print("[INFO] Related works string found for initial review context.")
        else:
            related_works_string = "No related works found for initial review context."
            print("[INFO] No related works found for initial review context.")
        return related_works_string

    def _build_review_prompt(self, text: str, related_works_string: str) -> str:
        # This uses the standard 'neurips_form' for the initial academic review.
        # It needs 'template_instructions' and 'related_works_string' placeholders.
        current_neurips_form = self.prompts.get('neurips_form', '{template_instructions}') # Fallback
        if '{related_works_string}' not in current_neurips_form and hasattr(self.prompts, 'template_instructions'):
             # If related_works_string is not in the main form, but template_instructions is, it might be nested.
             # This assumes template_instructions itself has the related_works_string placeholder.
             current_template_instructions = self.prompts.template_instructions.format(related_works_string=related_works_string)
             base_prompt = current_neurips_form.format(template_instructions=current_template_instructions)
        elif '{related_works_string}' in current_neurips_form and hasattr(self.prompts, 'template_instructions'):
            # If both are top-level in neurips_form (which is unusual based on previous structure)
             base_prompt = current_neurips_form.format(template_instructions=self.prompts.template_instructions, related_works_string=related_works_string)
        elif '{related_works_string}' in current_neurips_form:
            # If only related_works_string is in the form (template_instructions might be static or missing)
            base_prompt = current_neurips_form.format(related_works_string=related_works_string)
        else:
            print("[WARNING] 'neurips_form' may not be correctly formatted with placeholders. Using it as is.")
            base_prompt = current_neurips_form

        return f"{base_prompt}\n\nHere is the paper you are asked to review for its academic merit:\n```text\n{text}\n```"

    def _generate_query(self, text: str) -> str:
        query_prompt_template = self.prompts.get('query_prompt', '')
        if not query_prompt_template:
            print("[WARNING] Query prompt template not found. Cannot generate search query for related works.")
            return ""
        query_prompt = query_prompt_template.format(paper_text=text)
        
        response, _ = get_response_from_llm(
            query_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.get('reviewer_system_prompt_neg', self.prompts.reviewer_system_prompt_base),
            temperature=self.temperature,
            msg_history=[],
        )
        query_data = extract_json_between_markers(response)
        return str(query_data.get("Query", "")) if query_data else ""

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_review( # Used for initial academic review
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
        )
        review = extract_json_between_markers(llm_review)
        return review if review is not None else {}, msg_history

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_review_academic( # Renamed from _reflect_review, for academic review reflection
        self,
        review: Dict[str, Any],
        reviewer_system_prompt: str,
        related_works_string: str,
        msg_history: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
        reflection_prompt_template = self.prompts.get('reviewer_reflection_prompt', '')
        if not reflection_prompt_template:
            print("[WARNING] Reviewer reflection prompt not found! Skipping reflection.")
            return review, msg_history, True # Return original review, done=True
        
        updated_prompt = (
            f"Previous academic review: {json.dumps(review)}\n"
            + reflection_prompt_template.format(related_works_string=related_works_string)
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
        is_done = "I am done" in text # Assuming this marker is still used

        return new_review or {}, msg_history, is_done

    # _write_meta_review from original is replaced by _write_final_meta_review
    # _aggregate_scores can be reused or adapted for the final_meta_review.
    def _aggregate_scores( # Used by _write_final_meta_review for academic scores
        self, meta_review: Dict[str, Any], reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # This aggregates academic scores. Ethical scores would be handled separately or added by LLM.
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
            # else: if no valid scores for a field from initial reviews, meta_review LLM output for that field is kept (if any)
        return meta_review

    @staticmethod
    def _format_paper_results(papers: List[Dict[str, Any]]) -> str:
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'No title')
            # Authors/Source might be in different fields depending on PaperSearchTool output
            authors = paper.get('authors', 'No authors') 
            if isinstance(authors, list):
                authors = ", ".join(authors)
            venue = paper.get('venue', paper.get('source', paper.get('info', 'No venue')))
            year = paper.get('year', '')
            paper_strings.append(
                f"{i}: {title} ({authors}, {year}). Venue: {venue}. Abstract: {paper.get('abstract', 'N/A')[:200]}..."
            )
        return "\n\n".join(paper_strings) 