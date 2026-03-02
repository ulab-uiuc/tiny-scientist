import json
import time
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, Runner
from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .tool_impls import PaperSearchTool
from .tools.agent_tools import build_research_tools
from .utils.agent_sdk import is_claude_agent_sdk, resolve_agent_sdk
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.input_formatter import InputFormatter
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from .utils.openai_skills import build_openai_skill_shell_tool
from .utils.pricing import estimate_prompt_cost, estimate_tokens_from_text
from .utils.rich_output import (
    print_cost_delta_summary,
    print_mapping_table,
    print_stage_progress,
    print_task_event,
    print_todo_table,
    summarize_review,
)
from .utils.sdk_mcp import claude_allowed_mcp_tools, ensure_mcp_config
from .utils.skill_loader import skill_instructions
from .utils.sdk_client import configure_openai_agents_for_model, track_sdk_cost


class _ReviewerLegacy:
    def __init__(
        self,
        model: str,
        tools: Optional[List[Any]] = None,
        num_reviews: int = 3,
        num_reflections: int = 2,
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
        cost_tracker: Optional[BudgetChecker] = None,
        pre_reflection_threshold: float = 0.5,
        post_reflection_threshold: float = 0.8,
        s2_api_key: Optional[str] = None,
    ):
        self.tools = tools or []
        self.num_reviews = num_reviews
        self.num_reflections = num_reflections
        self.client, self.model = create_client(model)
        self.temperature = temperature
        self.config = Config(prompt_template_dir)
        self.searcher: PaperSearchTool = PaperSearchTool(
            s2_api_key=s2_api_key, engine="semanticscholar"
        )
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_related_works_string = ""
        self.last_related_works: List[Dict[str, Any]] = []
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.pre_reflection_threshold = pre_reflection_threshold
        self.post_reflection_threshold = post_reflection_threshold

        self.prompts = self.config.prompt_template.reviewer_prompt
        self.prompts.neurips_form = self.prompts.neurips_form.format(
            template_instructions=self.prompts.template_instructions
        )

    def review(
        self, pdf_path: Optional[str] = None, tex_path: Optional[str] = None
    ) -> str:
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

        query = self._generate_query(paper_text)

        related_works_string = self._get_related_works(query)
        self.last_related_works_string = related_works_string

        base_prompt = self._build_review_prompt(paper_text, related_works_string)
        system_prompt = self.prompts.reviewer_system_prompt_neg

        review, _ = self._generate_review(base_prompt, system_prompt, msg_history=[])
        if isinstance(review, dict):
            review["ReferenceLinks"] = self._extract_reference_links(
                self.last_related_works
            )
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

    def run(
        self, pdf_path: Optional[str] = None, tex_path: Optional[str] = None
    ) -> Dict[str, Any]:
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
            self.last_related_works = related_papers
            related_works_string = self._format_paper_results(related_papers)
            print("✅Related Works String Found")
        else:
            self.last_related_works = []
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
        if isinstance(meta_review, dict):
            meta_review["ReferenceLinks"] = self._extract_reference_links(
                self.last_related_works
            )

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
            source = paper.get("authors") or paper.get("source") or "No authors"
            venue = paper.get("venue", "")
            year = paper.get("year", "")
            info = paper.get("info") or f"{venue} {year}".strip() or "No venue"
            abstract = paper.get("abstract", "")
            url = str(paper.get("url") or "").strip()

            # Format: Title. Authors. Venue.\nAbstract: ...
            paper_str = f"{i}: {title}. {source}. {info}"
            if url:
                paper_str += f"\nURL: {url}"
            if abstract and len(abstract.strip()) > 0:
                # Truncate very long abstracts
                abstract_text = abstract.strip()
                if len(abstract_text) > 500:
                    abstract_text = abstract_text[:500] + "..."
                paper_str += f"\nAbstract: {abstract_text}"

            paper_strings.append(paper_str)

        return "\n\n".join(paper_strings)

    @staticmethod
    def _extract_reference_links(papers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        seen: set[str] = set()
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            url = str(paper.get("url") or paper.get("source") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            links.append(
                {
                    "title": str(paper.get("title", "Untitled")),
                    "url": url,
                    "source_type": str(paper.get("source_type", "paper_search")),
                }
            )
        return links


class Reviewer(_ReviewerLegacy):
    """Reviewer variant with configurable OpenAI/Claude agent SDK backends."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        agent_sdk = kwargs.pop("agent_sdk", None)
        use_claude_agent_sdk = kwargs.pop("use_claude_agent_sdk", None)
        super().__init__(*args, **kwargs)
        self.agent_sdk = resolve_agent_sdk(
            agent_sdk=agent_sdk,
            use_claude_agent_sdk=use_claude_agent_sdk,
        )
        self._sdk_cwd = "."
        self._shared_tools: List[Any] = []
        self._mcp_config_path: Optional[str] = None
        self._review_tool_policy = (
            "Tool policy: use web_search for recent claims, paper_search for related "
            "academic work, and code_search for implementation comparison."
        )
        self.review_agent = None
        if is_claude_agent_sdk(self.agent_sdk):
            self._setup_claude_sdk()
        else:
            self._setup_openai_sdk()

    def _setup_openai_sdk(self) -> None:
        configure_openai_agents_for_model(self.model)
        self._shared_tools = build_research_tools(
            model=self.model, include_drawer=False
        )
        skill_shell_tool = build_openai_skill_shell_tool(
            stage="reviewer",
            working_directory=self._sdk_cwd,
        )
        if skill_shell_tool is not None:
            self._shared_tools.append(skill_shell_tool)
            self._review_tool_policy += (
                " Use the shell tool when a mounted OpenAI skill is relevant."
            )

        self.review_agent = Agent(
            name="PaperReviewer",
            instructions=skill_instructions(
                "reviewer",
                f"{self.prompts.reviewer_system_prompt_neg}\n\n{self._review_tool_policy}",
            ),
            tools=self._shared_tools,
            model=self.model,
        )

    def _setup_claude_sdk(self) -> None:
        self._mcp_config_path = ensure_mcp_config(self._sdk_cwd, include_drawer=False)
        self.review_agent = self._make_runtime_agent(
            name="PaperReviewer",
            instructions=f"{self.prompts.reviewer_system_prompt_neg}\n\n"
            "Tool policy: use MCP research tools for recent claims and supporting context.",
        )

    def _make_runtime_agent(
        self,
        name: str,
        instructions: str,
        use_research_tools: bool = True,
    ) -> Any:
        if is_claude_agent_sdk(self.agent_sdk):
            from .utils.claude_agent_runner import ClaudeAgentRunner

            allowed_tools = (
                ["Skill", *claude_allowed_mcp_tools(include_drawer=False)]
                if use_research_tools
                else ["Skill"]
            )
            return ClaudeAgentRunner(
                instructions=instructions,
                allowed_tools=allowed_tools,
                cwd=self._sdk_cwd,
                permission_mode="bypassPermissions",
                cost_tracker=self.cost_tracker,
                model=self.model,
                mcp_config_path=self._mcp_config_path,
            )

        return Agent(
            name=name,
            instructions=skill_instructions("reviewer", instructions),
            tools=self._shared_tools if use_research_tools else [],
            model=self.model,
        )

    def _build_todo(self) -> List[Dict[str, Any]]:
        todo: List[Dict[str, Any]] = [
            {
                "step": 1,
                "action": "generate_review",
                "name": "generate_review",
                "description": "Generate a structured review grounded in the paper content and retrieved context.",
            }
        ]
        next_step = 2
        if self.tools:
            todo.append(
                {
                    "step": next_step,
                    "action": "apply_tools",
                    "name": "apply_tools",
                    "description": "Apply available review tools to enrich or validate the review draft.",
                }
            )
            next_step += 1
        if self.num_reflections > 0:
            todo.append(
                {
                    "step": next_step,
                    "action": "reflect_review",
                    "name": "reflect_review",
                    "description": "Run reviewer reflection rounds to tighten judgments and improve consistency.",
                }
            )
            next_step += 1
        todo.append(
            {
                "step": next_step,
                "action": "meta_review",
                "name": "meta_review",
                "description": "Aggregate individual reviews into a final meta review.",
            }
        )
        return todo

    def _apply_tools_to_review(self, review_json: str) -> str:
        current_review = review_json
        for tool in self.tools:
            tool_input = json.dumps({"review": current_review})
            tool_output = tool.run(tool_input)
            if "review" in tool_output:
                current_review = tool_output["review"]["review"]
        return current_review

    def _apply_reflections(self, review_json: str) -> str:
        current_review = review_json
        budget = self.cost_tracker.get_budget()
        if (
            budget is not None
            and self.cost_tracker.get_total_cost() / budget >= self.pre_reflection_threshold
        ):
            print("[Reviewer] Skipping review reflections due to budget limit.")
            return current_review

        max_rounds = self.num_reflections
        estimated_rounds = self._estimate_usable_reflection_rounds(current_review)
        if estimated_rounds is not None:
            if estimated_rounds <= 0:
                print("[Reviewer] Estimated remaining budget insufficient for reflections.")
                return current_review
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
        return current_review

    def run(
        self, pdf_path: Optional[str] = None, tex_path: Optional[str] = None
    ) -> Dict[str, Any]:
        todo = self._build_todo()
        print_todo_table("Reviewer", todo)
        execution_todo = [
            item for item in todo if str(item.get("action", "")) != "meta_review"
        ]

        all_reviews: List[Dict[str, Any]] = []
        for i in range(self.num_reviews):
            print(f"[TODO][Reviewer] Processing review {i + 1}/{self.num_reviews}")
            current_review = ""
            for idx, item in enumerate(execution_todo, start=1):
                action = str(item.get("action", ""))
                before_total, before_tasks = self.cost_tracker.snapshot()
                before_global_total, before_global_tasks = self.cost_tracker.global_snapshot()
                print_stage_progress(
                    f"Reviewer Progress ({i + 1}/{self.num_reviews})",
                    idx,
                    len(execution_todo),
                    str(item.get("name", action)),
                )
                if action == "generate_review":
                    current_review = self.review(pdf_path=pdf_path, tex_path=tex_path)
                    self._print_review_step_summary(
                        f"Reviewer Draft ({i + 1}/{self.num_reviews})",
                        current_review,
                    )
                elif action == "apply_tools":
                    current_review = self._apply_tools_to_review(current_review)
                    self._print_review_step_summary(
                        f"Reviewer Tools ({i + 1}/{self.num_reviews})",
                        current_review,
                    )
                elif action == "reflect_review":
                    current_review = self._apply_reflections(current_review)
                    self._print_review_step_summary(
                        f"Reviewer Reflection ({i + 1}/{self.num_reviews})",
                        current_review,
                    )
                print_stage_progress(
                    f"Reviewer Progress ({i + 1}/{self.num_reviews})",
                    idx,
                    len(execution_todo),
                    str(item.get("name", action)),
                    status="done",
                )
                after_total, after_tasks = self.cost_tracker.snapshot()
                after_global_total, after_global_tasks = self.cost_tracker.global_snapshot()
                print_cost_delta_summary(
                    f"Reviewer Cost: {action}",
                    before_total,
                    before_tasks,
                    after_total,
                    after_tasks,
                    global_before_total=before_global_total,
                    global_before_tasks=before_global_tasks,
                    global_after_total=after_global_total,
                    global_after_tasks=after_global_tasks,
                )
            try:
                parsed = json.loads(current_review)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                all_reviews.append(parsed)

        print_stage_progress(
            "Reviewer Progress (meta)",
            len(execution_todo),
            len(execution_todo),
            "Aggregate Meta Review",
        )
        before_total, before_tasks = self.cost_tracker.snapshot()
        before_global_total, before_global_tasks = self.cost_tracker.global_snapshot()
        result = self._write_meta_review(all_reviews)
        self._print_review_step_summary("Reviewer Meta Review", result)
        after_total, after_tasks = self.cost_tracker.snapshot()
        after_global_total, after_global_tasks = self.cost_tracker.global_snapshot()
        print_cost_delta_summary(
            "Reviewer Cost: meta_review",
            before_total,
            before_tasks,
            after_total,
            after_tasks,
            global_before_total=before_global_total,
            global_before_tasks=before_global_tasks,
            global_after_total=after_global_total,
            global_after_tasks=after_global_tasks,
        )
        print_stage_progress(
            "Reviewer Progress (meta)",
            len(execution_todo),
            len(execution_todo),
            "Aggregate Meta Review",
            status="done",
        )
        return result

    def _print_review_step_summary(
        self, title: str, review_payload: Any
    ) -> None:
        review_obj: Dict[str, Any] = {}
        if isinstance(review_payload, dict):
            review_obj = review_payload
        elif isinstance(review_payload, str):
            try:
                parsed = json.loads(review_payload)
                if isinstance(parsed, dict):
                    review_obj = parsed
            except json.JSONDecodeError:
                review_obj = {}
        if not review_obj:
            return
        print_mapping_table(title, summarize_review(review_obj))

    def _run_sdk_call(self, agent: Any, prompt: str, task_name: str) -> str:
        start = time.perf_counter()
        print_task_event("Reviewer", task_name, "START")
        if self.agent_sdk == "claude":
            result = agent.run_sync(prompt, task_name)
            print_task_event("Reviewer", task_name, "END", time.perf_counter() - start)
            return result
        result = Runner.run_sync(agent, prompt)
        track_sdk_cost(result, self.cost_tracker, self.model, task_name)
        print_task_event("Reviewer", task_name, "END", time.perf_counter() - start)
        return result.final_output or ""

    def _generate_query(self, text: str) -> str:
        query_prompt = self.prompts.query_prompt.format(paper_text=text)
        response = self._run_sdk_call(self.review_agent, query_prompt, "generate_query")
        query_data = extract_json_between_markers(response)
        return str(query_data.get("Query", "")) if query_data else ""

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_review(
        self,
        base_prompt: str,
        reviewer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if reviewer_system_prompt == self.prompts.reviewer_system_prompt_neg:
            agent = self.review_agent
        else:
            agent = self._make_runtime_agent(
                name="PaperReviewerCustom",
                instructions=reviewer_system_prompt,
            )

        llm_review = self._run_sdk_call(agent, base_prompt, "generate_review")
        review = extract_json_between_markers(llm_review)
        return review if review is not None else {}, []

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

        if reviewer_system_prompt == self.prompts.reviewer_system_prompt_neg:
            agent = self.review_agent
        else:
            agent = self._make_runtime_agent(
                name="PaperReviewerCustom",
                instructions=reviewer_system_prompt,
            )

        text = self._run_sdk_call(agent, updated_prompt, "reflect_review")

        new_review = extract_json_between_markers(text)
        is_done = "I am done" in text
        return new_review or {}, [], is_done

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

        meta_agent = self._make_runtime_agent(
            name="MetaReviewer",
            instructions=meta_system_prompt,
        )
        llm_meta_review = self._run_sdk_call(meta_agent, meta_prompt, "write_meta_review")

        meta_review = extract_json_between_markers(llm_meta_review)
        if meta_review is None:
            return {}
        if isinstance(meta_review, dict):
            meta_review["ReferenceLinks"] = self._extract_reference_links(
                self.last_related_works
            )
        return self._aggregate_scores(meta_review, reviews)
