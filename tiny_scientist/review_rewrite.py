import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from rich import print

from .configs import Config
from .tool import BaseTool, PaperSearchTool
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)

class ReviewRewriter:
    def __init__(
        self,
        model: str,
        tools: List[BaseTool],
        num_reviews: int = 1,
        num_reflections: int = 1,
        temperature: float = 0,
        prompt_template_dir: Optional[str] = None,
    ):
        self.tools = tools
        self.num_reviews = num_reviews
        self.num_reflections = num_reflections
        self.client, self.model = create_client(model)
        self.temperature = temperature
        self.config = Config(prompt_template_dir)
        self.searcher = PaperSearchTool()
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_related_works_string = ""

        self.prompts = self.config.prompt_template.reviewer_prompt
        # DEBUG: Print available attributes in self.prompts
        print(f"[DEBUG] ReviewRewriter.__init__: Attributes in self.prompts: {dir(self.prompts)}")
        # DEBUG: Check for specific problematic prompts
        print(f"[DEBUG] Has ethical_review_guidelines_prompt: {hasattr(self.prompts, 'ethical_review_guidelines_prompt')}")
        print(f"[DEBUG] Has rewrite_paper_instruction_prompt: {hasattr(self.prompts, 'rewrite_paper_instruction_prompt')}")

        if hasattr(self.prompts, 'neurips_form') and hasattr(self.prompts, 'template_instructions'):
            self.prompts.neurips_form = self.prompts.neurips_form.format(
                template_instructions=self.prompts.template_instructions
            )
        else:
            print("[WARNING] NeurIPS form or template instructions not found in prompts. Standard review might be affected.")

    def _perform_ethical_review(self, paper_text: str) -> Dict[str, Any]:
        print("[INFO] Performing detailed ethical review...")
        # DEBUG: Check self.prompts again before access
        print(f"[DEBUG] ReviewRewriter._perform_ethical_review: Attributes in self.prompts: {dir(self.prompts)}")
        print(f"[DEBUG] Type of self.prompts: {type(self.prompts)}")
        print(f"[DEBUG] Value of self.prompts.ethical_review_guidelines_prompt (first 50 chars if exists): {str(getattr(self.prompts, 'ethical_review_guidelines_prompt', 'NOT FOUND'))[:50]}")

        ethical_review_prompt_template = getattr(self.prompts, 'ethical_review_guidelines_prompt', '')
        if not ethical_review_prompt_template:
            print("[ERROR] Ethical review guidelines prompt not found!")
            return {"error": "Ethical review guidelines prompt missing."}

        ethical_review_prompt = ethical_review_prompt_template.format(paper_text=paper_text)
        system_prompt = getattr(self.prompts, 'ethical_reviewer_system_prompt', getattr(self.prompts, 'reviewer_system_prompt_base', ''))

        response, _ = get_response_from_llm(
            msg=ethical_review_prompt,
            client=self.client,
            model=self.model,
            system_message=system_prompt,
            temperature=self.temperature,
        )
        
        ethical_review_data = extract_json_between_markers(response)
        if ethical_review_data:
            print("[INFO] Ethical review data extracted.")
            return ethical_review_data
        else:
            print("[WARNING] Could not extract structured ethical review data. Returning raw response.")
            return {"raw_ethical_review": response}

    def rewrite_paper_based_on_ethical_feedback(self, paper_text: str, ethical_review_feedback: Dict[str, Any]) -> str:
        print("[INFO] Rewriting paper based on ethical feedback...")
        # DEBUG: Check self.prompts before access for rewrite prompt
        print(f"[DEBUG] ReviewRewriter.rewrite_paper_based_on_ethical_feedback: Attributes in self.prompts: {dir(self.prompts)}")
        print(f"[DEBUG] Value of self.prompts.rewrite_paper_instruction_prompt (first 50 chars if exists): {str(getattr(self.prompts, 'rewrite_paper_instruction_prompt', 'NOT FOUND'))[:50]}")

        rewrite_prompt_template = getattr(self.prompts, 'rewrite_paper_instruction_prompt', '')
        if not rewrite_prompt_template:
            print("[ERROR] Rewrite paper instruction prompt not found!")
            return paper_text

        feedback_str = json.dumps(ethical_review_feedback, indent=2)
        rewrite_prompt = rewrite_prompt_template.format(paper_text=paper_text, ethical_feedback=feedback_str)
        system_prompt = getattr(self.prompts, 'rewrite_paper_system_prompt', getattr(self.prompts, 'reviewer_system_prompt_base', ''))

        rewritten_text, _ = get_response_from_llm(
            msg=rewrite_prompt,
            client=self.client,
            model=self.model,
            system_message=system_prompt,
            temperature=self.temperature,
        )
        if rewritten_text:
            print("[INFO] Paper rewritten based on ethical feedback.")
            return rewritten_text
        else:
            print("[WARNING] Paper rewriting failed or returned empty. Returning original text.")
            return paper_text

    def _write_final_meta_review(self, paper_text: str, initial_reviews: List[Dict[str, Any]], ethical_review: Dict[str, Any], rewritten_paper_text: Optional[str] = None) -> Dict[str, Any]:
        print("[INFO] Writing final meta-review...")
        if not initial_reviews and not ethical_review:
            print("[WARNING] No initial academic reviews provided for meta-review. Meta-review will be based on ethical review and paper text.")

        formatted_initial_reviews = "".join(
            f"\nInitial Review {i + 1}:\n```json\n{json.dumps(r)}\n```\n"
            for i, r in enumerate(initial_reviews)
        ) if initial_reviews else "No initial academic reviews conducted."
        
        formatted_ethical_review = f"\nEthical Review:\n```json\n{json.dumps(ethical_review)}\n```\n"

        meta_review_prompt_template = getattr(self.prompts, 'final_meta_review_prompt', getattr(self.prompts, 'neurips_form', ''))
        final_paper_text_for_meta_review = rewritten_paper_text if rewritten_paper_text else paper_text

        meta_prompt_args = {
            "paper_text": final_paper_text_for_meta_review,
            "initial_reviews_summary": formatted_initial_reviews,
            "ethical_review_summary": formatted_ethical_review,
            "related_works_string": self.last_related_works_string or "No related works information available."
        }
        try:
            meta_prompt = meta_review_prompt_template.format(**meta_prompt_args)
        except KeyError as e:
            print(f"[ERROR] KeyError formatting meta_review_prompt_template: {e}. Template: {meta_review_prompt_template[:100]}... Args: {list(meta_prompt_args.keys())}")
            return {"error": f"Failed to format meta review prompt due to missing key: {e}"}

        meta_system_prompt_template = getattr(self.prompts, 'meta_reviewer_system_prompt', getattr(self.prompts, 'reviewer_system_prompt_base', ''))
        try:
            meta_system_prompt = meta_system_prompt_template.format(reviewer_count=len(initial_reviews) if initial_reviews else 0)
        except KeyError as e:
            print(f"[WARNING] KeyError formatting meta_system_prompt: {e}. Using template as is: {meta_system_prompt_template[:100]}...")
            meta_system_prompt = meta_system_prompt_template

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
            final_meta_review_data = {"raw_final_meta_review": llm_meta_review}

        if initial_reviews:
            final_meta_review_data = self._aggregate_scores(final_meta_review_data, initial_reviews)

        print("[INFO] Final meta-review generated.")
        return final_meta_review_data

    def _initial_academic_review(self, paper_text: str) -> str:
        print(f"[INFO] Generating initial academic review for provided paper text (length: {len(paper_text)} chars).")
        if not paper_text:
            raise ValueError("No paper text provided for initial academic review.")

        query = self._generate_query(paper_text)
        related_works_string = self._get_related_works(query)
        self.last_related_works_string = related_works_string

        base_prompt = self._build_review_prompt(paper_text, related_works_string)
        system_prompt = getattr(self.prompts, 'reviewer_system_prompt_neg', getattr(self.prompts, 'reviewer_system_prompt_base', ''))

        review_data, _ = self._generate_review(base_prompt, system_prompt, msg_history=[])
        return json.dumps(review_data, indent=2)

    def _reflect_initial_review(self, review_json: str, paper_text: str) -> str:
        current_review = json.loads(review_json)
        if not current_review:
            raise ValueError("No initial review provided for reflection.")

        system_prompt = getattr(self.prompts, 'reviewer_system_prompt_neg', getattr(self.prompts, 'reviewer_system_prompt_base', ''))
        related_works_string = self.last_related_works_string
        if not related_works_string:
            print("[INFO] Regenerating related works string for reflection context as it was empty.")
            query = self._generate_query(paper_text)
            related_works_string = self._get_related_works(query)
            self.last_related_works_string = related_works_string

        new_review, _, _ = self._reflect_review_academic(
            review=current_review,
            reviewer_system_prompt=system_prompt,
            related_works_string=related_works_string,
            paper_text_for_reflection=paper_text,
            msg_history=[],
        )
        return json.dumps(new_review, indent=2)

    def run(self, original_paper_text: str) -> Dict[str, Any]:
        if not original_paper_text:
            print("[ERROR] No paper text provided to ReviewRewriter.")
            return {"error": "No paper text provided."}

        print(f"[INFO] Starting Review-Rewrite process for provided text (length: {len(original_paper_text)} chars).")

        # Check environment variable to decide on ethical review
        perform_ethical_review_env = os.environ.get("PERFORM_ETHICAL_REVIEW", "true").lower()
        should_perform_ethical_review = perform_ethical_review_env == "true"

        print(f"[INFO] PERFORM_ETHICAL_REVIEW environment variable: '{perform_ethical_review_env}'")
        print(f"[INFO] Should perform ethical review and rewrite: {should_perform_ethical_review}")

        all_initial_reviews_json = []
        current_academic_review_str = "{}"
        for i in range(self.num_reviews):
            print(f"[INFO] Generating initial academic review {i + 1}/{self.num_reviews}")
            current_academic_review_str = self._initial_academic_review(original_paper_text)
            
            try:
                temp_review_dict = json.loads(current_academic_review_str)
            except json.JSONDecodeError:
                print(f"[ERROR] Initial academic review {i+1} was not valid JSON: {current_academic_review_str[:200]}...")
                temp_review_dict = {}

            for tool in self.tools:
                tool_input_dict = {"review": temp_review_dict}
                tool_output = tool.run(json.dumps(tool_input_dict))
                if isinstance(tool_output, dict) and "review" in tool_output and "review" in tool_output["review"]:
                    temp_review_dict = tool_output["review"]["review"]
                elif isinstance(tool_output, str):
                    try:
                        temp_review_dict = json.loads(tool_output)
                    except json.JSONDecodeError:
                        print(f"[WARNING] Tool {tool.name if hasattr(tool, 'name') else 'Unknown Tool'} returned non-JSON string.")
            current_academic_review_str = json.dumps(temp_review_dict)

            for j in range(self.num_reflections):
                print(f"[INFO] Reflecting on initial academic review {i + 1} (Reflection {j + 1}/{self.num_reflections})")
                current_academic_review_str = self._reflect_initial_review(current_academic_review_str, original_paper_text)
            
            try:
                all_initial_reviews_json.append(json.loads(current_academic_review_str))
            except json.JSONDecodeError:
                print(f"[ERROR] Reflected academic review {i+1} was not valid JSON: {current_academic_review_str[:200]}...")
                all_initial_reviews_json.append({"error": "Invalid JSON for reflected review"})

        ethical_review_output: Dict[str, Any]
        rewritten_paper_text_str: str

        if should_perform_ethical_review:
            ethical_review_output = self._perform_ethical_review(original_paper_text)
            rewritten_paper_text_str = self.rewrite_paper_based_on_ethical_feedback(original_paper_text, ethical_review_output)
        else:
            print("[INFO] Skipping ethical review and paper rewrite based on PERFORM_ETHICAL_REVIEW environment variable.")
            ethical_review_output = {
                "status": "Ethical review skipped due to PERFORM_ETHICAL_REVIEW setting.",
                "EthicalReviewOverallSummary": "Ethical review was not performed for this paper."
                # Add other keys expected by _write_final_meta_review with default/skipped values if necessary
            }
            rewritten_paper_text_str = original_paper_text # Use original paper if rewrite is skipped

        final_meta_report = self._write_final_meta_review(
            paper_text=original_paper_text, # Meta review should ideally see the *original* paper for context if rewrite is skipped
                                            # Or pass rewritten_paper_text_str, which would be original if skipped.
            initial_reviews=all_initial_reviews_json,
            ethical_review=ethical_review_output, # This will be the skipped status or actual review
            rewritten_paper_text=rewritten_paper_text_str 
        )

        print("[INFO] Review-Rewrite process completed.")
        return {
            "original_paper_text_provided_length": len(original_paper_text),
            "initial_academic_reviews": all_initial_reviews_json,
            "ethical_review": ethical_review_output, # Will contain actual review or skipped status
            "rewritten_paper_content": rewritten_paper_text_str,
            "final_meta_review": final_meta_report,
            "ethical_review_performed": should_perform_ethical_review # Add a flag indicating if it was done
        }

    def _get_related_works(self, query: str) -> str:
        if not query:
            print("[INFO] Empty query for related works. Skipping search.")
            return "No related works query generated."
            
        if query in self._query_cache:
            related_papers = self._query_cache[query]
        else:
            try:
                results_dict = self.searcher.run(query)
                related_papers = list(results_dict.values()) if results_dict else []
                self._query_cache[query] = related_papers
            except Exception as e:
                print(f"[ERROR] PaperSearchTool failed for query '{query}': {e}")
                related_papers = []
                self._query_cache[query] = []

        if related_papers:
            related_works_string = self._format_paper_results(related_papers)
            print("[INFO] Related works string found for initial review context.")
        else:
            related_works_string = "No related works found for the query."
            print("[INFO] No related works found for initial review context based on the query.")
        return related_works_string

    def _build_review_prompt(self, text: str, related_works_string: str) -> str:
        current_neurips_form = getattr(self.prompts, 'neurips_form', '{template_instructions}')
        template_instructions_content = getattr(self.prompts, 'template_instructions', '{related_works_string}')
        
        try:
            formatted_template_instructions = template_instructions_content.format(related_works_string=related_works_string)
        except KeyError:
            print("[WARNING] 'related_works_string' placeholder not found in 'template_instructions'. Using template_instructions as is.")
            formatted_template_instructions = template_instructions_content
        except Exception as e:
            print(f"[ERROR] Could not format template_instructions: {e}. Using as is.")
            formatted_template_instructions = template_instructions_content

        try:
            args_for_neurips_form = {"template_instructions": formatted_template_instructions}
            if '{related_works_string}' in current_neurips_form and 'related_works_string' not in args_for_neurips_form:
                args_for_neurips_form['related_works_string'] = related_works_string
            
            expected_keys_neurips = re.findall(r'\{([^}]+)\}', current_neurips_form)
            final_args_neurips = {}
            for k_expected in expected_keys_neurips:
                if k_expected in args_for_neurips_form:
                    final_args_neurips[k_expected] = args_for_neurips_form[k_expected]
                else:
                    print(f"[WARNING] Key '{k_expected}' expected by neurips_form was not prepared. Using placeholder for '{k_expected}'.")
                    final_args_neurips[k_expected] = f"[Placeholder for {k_expected}]"

            base_prompt = current_neurips_form.format(**final_args_neurips)
        except KeyError as e:
            print(f"[ERROR] KeyError formatting neurips_form: {e}. Form: {current_neurips_form[:100]}... Provided args: {list(final_args_neurips.keys())}")
            base_prompt = current_neurips_form
        except Exception as e:
            print(f"[ERROR] Could not format neurips_form: {e}. Using as is.")
            base_prompt = current_neurips_form

        return f"{base_prompt}\n\nHere is the paper you are asked to review for its academic merit:\n```text\n{text}\n```"

    def _generate_query(self, text: str) -> str:
        query_prompt_template = getattr(self.prompts, 'query_prompt', '')
        if not query_prompt_template:
            print("[WARNING] Query prompt template not found. Cannot generate search query for related works.")
            return ""
        
        try:
            query_prompt = query_prompt_template.format(paper_text=text)
        except KeyError:
            print("[ERROR] 'paper_text' placeholder missing in query_prompt template. Cannot generate query.")
            return ""

        response, _ = get_response_from_llm(
            query_prompt,
            client=self.client,
            model=self.model,
            system_message=getattr(self.prompts, 'reviewer_system_prompt_neg', getattr(self.prompts, 'reviewer_system_prompt_base', '')),
            temperature=self.temperature,
            msg_history=[],
        )
        query_data = extract_json_between_markers(response)
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
        )
        review = extract_json_between_markers(llm_review)
        return review if review is not None else {}, msg_history

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _reflect_review_academic(
        self,
        review: Dict[str, Any],
        reviewer_system_prompt: str,
        related_works_string: str,
        paper_text_for_reflection: str,
        msg_history: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
        reflection_prompt_template = getattr(self.prompts, 'reviewer_reflection_prompt', '')
        if not reflection_prompt_template:
            print("[WARNING] Reviewer reflection prompt not found! Skipping reflection.")
            return review, msg_history, True

        reflection_prompt_args = {
            "related_works_string": related_works_string,
            "paper_text": paper_text_for_reflection,
            "previous_review": json.dumps(review)
        }
        
        expected_keys_reflection = re.findall(r'\{([^}]+)\}', reflection_prompt_template)
        final_args_reflection = {}
        for k_expected in expected_keys_reflection:
            if k_expected in reflection_prompt_args:
                final_args_reflection[k_expected] = reflection_prompt_args[k_expected]
            else:
                print(f"[WARNING] Key '{k_expected}' expected by reflection_prompt was not prepared. Using placeholder.")
                final_args_reflection[k_expected] = f"[Placeholder for {k_expected}]"

        try:
            formatted_reflection_guidance = reflection_prompt_template.format(**final_args_reflection)
        except KeyError as e:
            print(f"[ERROR] KeyError formatting reflection_prompt: {e}. Template: {reflection_prompt_template[:100]}... Args: {list(final_args_reflection.keys())}")
            return review, msg_history, True

        updated_prompt = formatted_reflection_guidance

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

        return new_review or {}, msg_history, is_done

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
        return meta_review

    @staticmethod
    def _format_paper_results(papers: List[Dict[str, Any]]) -> str:
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'No title')
            authors_data = paper.get('authors', [])
            authors = ", ".join(authors_data) if isinstance(authors_data, list) else str(authors_data)
            
            venue = paper.get('venue', paper.get('source', paper.get('journal', {}).get('name', 'No venue')))
            year = paper.get('year', '')
            abstract = paper.get('abstract', 'N/A')
            
            parts = [
                f"{i+1}: {str(title)} ({str(authors)}, {str(year)}).",
                f"Venue: {str(venue)}.",
                f"Abstract: {str(abstract)[:200]}..."
            ]
            paper_strings.append(" ".join(parts))
            
        return "\n\n".join(paper_strings) 