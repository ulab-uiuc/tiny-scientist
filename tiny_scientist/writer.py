import json
import os
import os.path as osp
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .tool import BaseTool, DrawerTool, PaperSearchTool
from .utils.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from .utils.output_formatter import (
    ACLOutputFormatter,
    BaseOutputFormatter,
    ICLROutputFormatter,
)
from .utils.pricing import estimate_prompt_cost, estimate_tokens_from_text

# ---- Constants --------------------------------------------------------------

REFINEMENT_SECTIONS_EXPERIMENTAL = [
    "Introduction",
    "Method",
    "Experimental_Setup",
    "Results",
    "Discussion",
    "Conclusion",
]

REFINEMENT_SECTIONS_NONEXP = [
    "Introduction",
    "Method",
    "Analysis",
    "Discussion",
    "Conclusion",
]

SECTION_ORDER_FOR_CONTEXT = [
    "Abstract",
    "Related_Work",
    "Introduction",
    "Method",
    "Experimental_Setup",
    "Results",
    "Discussion",
    "Conclusion",
]

MAX_ABSTRACT_SNIPPET = 400
MAX_CONTEXT_SECTION_LEN = 2000
SLEEP_SEARCH = 0.8
SLEEP_REFINE = 0.5


class Writer:
    """
    End-to-end paper writer with:
      - Section generation (intro/method/etc.)
      - Related work + citations via Semantic Scholar
      - Multi-round refinement and title polishing
      - Formatting to PDF via ACL/ICLR formatters
    """

    def __init__(
        self,
        model: str,
        output_dir: str,
        template: str,
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
        cost_tracker: Optional[BudgetChecker] = None,
        s2_api_key: Optional[str] = None,
        num_refinement_rounds: int = 2,
    ) -> None:
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.template = template
        self.temperature = temperature
        self.num_refinement_rounds = num_refinement_rounds

        # API key
        s2_api_key = s2_api_key or os.environ.get("S2_API_KEY")

        # Tools
        self.searcher: BaseTool = PaperSearchTool(
            s2_api_key=s2_api_key,
            engine="semanticscholar",
        )
        self.drawer: BaseTool = DrawerTool(model, prompt_template_dir, temperature)

        # Formatter selection
        if self.template == "acl":
            self.formatter: BaseOutputFormatter = ACLOutputFormatter(
                model=self.model, client=self.client
            )
        elif self.template == "iclr":
            self.formatter = ICLROutputFormatter(model=self.model, client=self.client)
        else:
            raise ValueError(f"Unknown template: {self.template!r}")

        # Prompts & config
        self.config = Config(prompt_template_dir)
        self.prompts = self.config.prompt_template.writer_prompt
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.system_prompt = self.prompts.write_system_prompt.format()

        # Runtime state
        self.generated_sections: Dict[str, str] = {}
        self.references: Dict[str, Any] = {}

    # ---- Public API ---------------------------------------------------------

    def run(
        self, idea: Dict[str, Any], experiment_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate a full paper and export a PDF.

        Returns:
            (output_pdf_path, paper_slug)
        """
        is_experimental = idea.get("is_experimental", True)
        code, experiment_result, baseline_result = self._load_experiment_bundle(
            is_experimental, experiment_dir
        )

        print("[step] Generating Related Work…")
        self._write_related_work(idea)  # non-fatal if it fails

        sections = (
            REFINEMENT_SECTIONS_EXPERIMENTAL
            if is_experimental
            else REFINEMENT_SECTIONS_NONEXP
        )
        for section in sections:
            self._write_section(idea, code, experiment_result, section, baseline_result)

        print("[step] Generating Abstract…")
        self._write_abstract(idea)

        print("[step] Refinement rounds…")
        self._refine_paper(num_rounds=self.num_refinement_rounds, add_citations=True)

        paper_name = self._slugify(idea.get("Title", "Research Paper"))
        output_pdf_path = f"{self.output_dir}/{paper_name}.pdf"

        self.formatter.run(
            content=self.generated_sections,
            references=self.references,
            output_dir=self.output_dir,
            output_pdf_path=output_pdf_path,
            name=self.generated_sections.get("Title", "Research Paper"),
        )
        self.cost_tracker.report()
        return output_pdf_path, paper_name

    # ---- Section generation -------------------------------------------------

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        """Generate abstract using all already-written sections as context."""
        title = idea.get("Title", "Research Paper")
        full_context = self._sections_as_markdown(exclude={"Abstract"})
        abstract_prompt = self.prompts.abstract_prompt.format(
            abstract_tips=self.prompts.section_tips.get("Abstract", ""),
            title=title,
            problem=idea.get("Problem", ""),
            importance=idea.get("Importance", ""),
            difficulty=idea.get("Difficulty", ""),
            novelty=idea.get("NoveltyComparison", ""),
            experiment=idea.get("Experiment", ""),
            full_paper_content=full_context,
        )
        abstract_content, _ = get_response_from_llm(
            msg=abstract_prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name="Abstract",
        )
        self.generated_sections["Abstract"] = abstract_content

    def _write_section(
        self,
        idea: Dict[str, Any],
        code: str,
        experiment_result: str,
        section: str,
        baseline_result: str = "",
    ) -> None:
        print(f"[section] {section}")
        ctx = self._build_section_context(
            idea, code, experiment_result, baseline_result
        )
        ctx["section_tips"] = self.prompts.section_tips.get(section, "")

        template = self.prompts.section_prompt.get(section)
        if not template:
            print(f"[warn] No prompt template for section: {section}")
            return

        prompt = template.format(**ctx)
        content, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"{section} section",
        )
        self.generated_sections[section] = content

        # Try citation enrichment (non-fatal)
        try:
            enriched = self._enrich_section_with_citations(
                section=section, idea=idea, max_queries=5
            )
            if enriched:
                self.generated_sections[section] = enriched
        except Exception as e:
            print(f"[warn] Citation enrichment failed in {section}: {e}")
            traceback.print_exc()

    # ---- Refinement & citations --------------------------------------------

    def _refine_paper(self, num_rounds: int = 2, add_citations: bool = True) -> None:
        """Multi-round refinement with optional citation passes."""
        remaining_budget = self._effective_remaining_budget()

        # Title refinement over full draft
        full_draft = self._sections_as_latex()
        title_prompt = self.prompts.title_refinement_prompt.format(
            full_draft=full_draft
        )

        if remaining_budget is not None:
            title_cost = estimate_prompt_cost(
                self.model,
                [self.system_prompt, title_prompt],
                expected_output_tokens=estimate_tokens_from_text(
                    self.generated_sections.get("Title", "")
                ),
            )
            if title_cost is not None and title_cost > remaining_budget:
                print(
                    "[Writer] Skipping refinement stage due to estimated budget constraints."
                )
                return
            if title_cost:
                remaining_budget -= title_cost

        refined_title, _ = get_response_from_llm(
            msg=title_prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name="Title Refinement",
        )
        self.generated_sections["Title"] = refined_title

        refinement_priority = [
            "Method",
            "Introduction",
            "Experimental_Setup",
            "Results",
            "Discussion",
            "Conclusion",  # Include but with special constraint to keep it 1 paragraph
        ]

        for r in range(1, num_rounds + 1):
            print(f"[refine] Round {r}/{num_rounds}")
            for section in refinement_priority:
                if section not in self.generated_sections:
                    continue

                other_ctx = self._other_sections_context(exclude={section})
                focus = (
                    "Add mathematical rigor, expand technical details, improve structure"
                    if r == 1
                    else "Deepen analysis, add design rationale, enhance clarity"
                    if r == 2
                    else "Polish writing, ensure coherence, strengthen arguments"
                )
                method_hint = (
                    "For Method: Add more \\paragraph{} blocks, equations, and technical depth"
                    if section == "Method"
                    else "Enhance technical detail and clarity"
                )

                prompt = self.prompts.multi_round_refinement_prompt.format(
                    section=section,
                    round_num=r,
                    total_rounds=num_rounds,
                    focus=focus,
                    section_content=self.generated_sections[section],
                    section_tips=self.prompts.section_tips.get(section, ""),
                    other_sections_context=other_ctx,
                    method_specific_instruction=method_hint,
                    error_list=self.prompts.error_list,
                )

                estimated_cost = None
                if remaining_budget is not None:
                    estimated_cost = estimate_prompt_cost(
                        self.model,
                        [self.system_prompt, prompt],
                        expected_output_tokens=estimate_tokens_from_text(
                            self.generated_sections.get(section, "")
                        ),
                    )
                    if estimated_cost is not None and estimated_cost > remaining_budget:
                        print(
                            f"[Writer] Skipping refinement for {section} in round {r} due to estimated budget constraints."
                        )
                        continue

                refined, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.system_prompt,
                    cost_tracker=self.cost_tracker,
                    task_name=f"Refine_R{r}_{section}",
                )
                self.generated_sections[section] = refined

                if remaining_budget is not None and estimated_cost:
                    remaining_budget -= estimated_cost

                if add_citations:
                    try:
                        idea_proxy = {"Title": self.generated_sections.get("Title", "")}
                        enriched = self._enrich_section_with_citations(
                            section, idea_proxy, max_queries=2
                        )
                        if enriched:
                            self.generated_sections[section] = enriched
                    except Exception as e:
                        print(
                            f"[warn] Citation add failed in round {r} for {section}: {e}"
                        )
                        traceback.print_exc()

                time.sleep(SLEEP_REFINE)

    def _effective_remaining_budget(self) -> Optional[float]:
        if hasattr(self.cost_tracker, "get_effective_remaining_budget"):
            return self.cost_tracker.get_effective_remaining_budget()
        return self.cost_tracker.get_remaining_budget()

    def _enrich_section_with_citations(
        self,
        section: str,
        idea: Dict[str, Any],
        max_queries: int = 4,
    ) -> Optional[str]:
        """Search relevant papers and ask LLM to weave citations into the section."""
        original = self.generated_sections.get(section, "")
        if not original or len(original) < 50:
            return None

        queries = self._generate_search_queries(
            idea,
            section=section,
            content_snippet=original[:600],
            max_queries=max_queries,
        )
        papers = self._search_papers_by_queries(queries)
        if not papers:
            return None

        print(f"[cite] {section}: found {len(papers)} candidates")

        # Merge into global references (dedupe by title)
        for title, pdata in papers.items():
            self.references.setdefault(title, pdata)

        paper_ctx = self._format_paper_context(papers)
        embed_prompt = self.prompts.add_new_citations_prompt.format(
            section=section,
            section_content=original,
            paper_context=paper_ctx,
            num_papers=len(papers),
        )
        enriched, _ = get_response_from_llm(
            msg=embed_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.citation_system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"Add_Citations_{section}",
        )
        return enriched

    # ---- Related Work -------------------------------------------------------

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        """Write Related Work from paper searches; store bibtex/abstracts as references."""
        try:
            queries = self._generate_search_queries(
                idea, section="Related_Work", max_queries=6
            )
            papers = self._search_papers_by_queries(queries)
            if not papers:
                print("[warn] No papers found for Related Work")
                return

            # Initial reference set is the related-work batch
            self.references = papers

            paper_ctx = self._format_paper_context(papers)
            experiment = idea.get("Experiment", "No experiment details provided")

            prompt = self.prompts.related_work_prompt.format(
                related_work_tips=self.prompts.section_tips["Related_Work"],
                experiment=experiment,
                references=paper_ctx,
            )
            content, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.write_system_prompt_related_work,
                cost_tracker=self.cost_tracker,
                task_name="Related_Work",
            )
            self.generated_sections["Related_Work"] = content
            print(f"[info] Related Work generated with {len(papers)} references")
        except Exception as e:
            print(f"[warn] Related Work generation failed: {e}")
            traceback.print_exc()

    # ---- Query/search utilities --------------------------------------------

    def _generate_search_queries(
        self,
        idea: Dict[str, Any],
        section: str = "",
        content_snippet: str = "",
        max_queries: int = 6,
    ) -> List[str]:
        """Ask LLM to propose queries; robust JSON parse with fallback."""
        title = self._to_text(idea.get("Title", ""))
        problem = self._to_text(idea.get("Problem", ""))
        novelty = self._to_text(idea.get("NoveltyComparison", ""))
        experiment = self._to_text(idea.get("Experiment", ""))

        prompt = self.prompts.citation_search_query_prompt.format(
            idea_title=title or "Research Paper",
            problem=problem,
            novelty=novelty,
            experiment=experiment,
            section=section or "General",
            snippet=content_snippet or "",
        )
        resp, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.citation_system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"generate_queries_{section or 'general'}",
        )

        queries: List[str] = []
        try:
            parsed = json.loads(resp)
        except json.JSONDecodeError:
            parsed = extract_json_between_markers(resp)

        if isinstance(parsed, list):
            for q in parsed:
                s = self._to_text(q)
                if s:
                    queries.append(s)

        return queries[:max_queries]

    def _search_papers_by_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Merge search results across queries keyed by paper title."""
        merged: Dict[str, Any] = {}
        for q in queries:
            try:
                results = self.searcher.run(q)
                if results:
                    for title, pdata in results.items():
                        merged.setdefault(title, pdata)
                time.sleep(SLEEP_SEARCH)
            except Exception as e:
                print(f"[error] Search failed for '{q}': {e}")
                traceback.print_exc()
        return merged

    # ---- Formatting helpers -------------------------------------------------

    def _format_paper_context(self, papers: Dict[str, Any]) -> str:
        """
        Provide compact entries with bibtex keys + short abstracts for the LLM.
        """
        entries: List[str] = []
        for title, pdata in papers.items():
            authors = self._format_authors(pdata.get("authors", ""))
            abstract = (pdata.get("abstract", "") or "")[:MAX_ABSTRACT_SNIPPET]
            year = pdata.get("year", "")
            venue = pdata.get("venue", "")
            bibtex = pdata.get("bibtex", "") or ""

            bibkey = "UNKNOWN"
            m = re.search(r"@\w+\{(.+?),", bibtex)
            if m:
                bibkey = m.group(1).strip()

            # Use format that doesn't conflict with .format() syntax
            entry = f"- [CITE_KEY: {bibkey}] {title} ({authors}, {venue}, {year})\n  Abstract: {abstract}"
            entries.append(entry)

        return "\n\n".join(entries)

    def _format_authors(self, authors: Any) -> str:
        """Accept strings, list[str], or list[dict{name}]."""
        if isinstance(authors, str) and authors.strip():
            return authors
        if isinstance(authors, list) and authors:
            if isinstance(authors[0], dict):
                names = [a.get("name", "") for a in authors if a.get("name")]
            else:
                names = [str(a) for a in authors if str(a).strip()]
            if not names:
                return "Unknown authors"
            return " and ".join(names) if len(names) <= 3 else f"{names[0]} et al."
        return "Unknown authors"

    # ---- Context & extraction ----------------------------------------------

    def _build_section_context(
        self,
        idea: Dict[str, Any],
        code: str,
        experiment_result: str,
        baseline_result: str,
    ) -> Dict[str, Any]:
        """Assemble all variables used by section prompt templates."""
        # Format the entire idea as a structured string
        idea_str = self._format_idea_as_string(idea)

        return {
            "idea": idea_str,  # Complete idea in formatted string
            "title": idea.get("Title", "Research Paper"),  # Keep for specific use
            "code": code,
            "experiment_results": experiment_result,
            "baseline_results": baseline_result,
            "previous_context": self._sections_as_markdown(
                exclude=set()
            ),  # All previous sections
        }

    def _format_idea_as_string(self, idea: Dict[str, Any]) -> str:
        """Format the complete idea as a readable string."""
        parts = []

        if idea.get("Title"):
            parts.append(f"**Title**: {idea['Title']}")
        if idea.get("Problem"):
            parts.append(f"\n**Research Problem**: {idea['Problem']}")
        if idea.get("Importance"):
            parts.append(f"\n**Importance**: {idea['Importance']}")
        if idea.get("Difficulty"):
            parts.append(f"\n**Difficulty**: {idea['Difficulty']}")
        if idea.get("NoveltyComparison"):
            parts.append(f"\n**Novelty Comparison**: {idea['NoveltyComparison']}")
        if idea.get("Approach"):
            parts.append(f"\n**Proposed Approach**: {idea['Approach']}")
        if idea.get("Experiment") or idea.get("ResearchPlan"):
            exp = idea.get("Experiment", idea.get("ResearchPlan", ""))
            parts.append(f"\n**Experiment Plan**: {exp}")

        return "".join(parts)

    def _other_sections_context(self, exclude: set) -> str:
        """Join other sections (truncated) for refinement prompts."""
        chunks: List[str] = []
        for sec in SECTION_ORDER_FOR_CONTEXT:
            if sec in exclude:
                continue
            content = self.generated_sections.get(sec)
            if not content:
                continue
            if len(content) > MAX_CONTEXT_SECTION_LEN:
                content = content[:MAX_CONTEXT_SECTION_LEN] + "\n[...truncated...]"
            chunks.append(f"## {sec}\n{content}")
        return "\n\n".join(chunks)

    def _sections_as_markdown(self, exclude: set = frozenset()) -> str:
        """Render all sections as markdown blocks for context prompts."""
        blocks = []
        for sec, content in self.generated_sections.items():
            if sec in exclude:
                continue
            blocks.append(f"## {sec}\n{content}")
        return "\n\n".join(blocks)

    def _sections_as_latex(self) -> str:
        """Render all sections roughly as LaTeX for title refinement."""
        return "\n\n".join(
            [
                f"\\section{{{sec}}}\n\n{cnt}"
                for sec, cnt in self.generated_sections.items()
            ]
        )

    # ---- I/O & small utilities --------------------------------------------

    def _load_experiment_bundle(
        self, is_experimental: bool, experiment_dir: Optional[str]
    ) -> Tuple[str, str, str]:
        """Read code/results files if required by experimental papers."""
        if not is_experimental:
            return "", "", ""
        if not experiment_dir:
            raise ValueError("Experimental papers require an experiment_dir")

        code = self._read_text(osp.join(experiment_dir, "experiment.py"))
        exp = self._read_text(osp.join(experiment_dir, "experiment_results.txt"))
        base = self._read_text(
            osp.join(experiment_dir, "baseline_results.txt"), missing_ok=True
        )
        return code, exp, base

    @staticmethod
    def _read_text(path: str, missing_ok: bool = False) -> str:
        """Read a UTF-8 text file; optionally allow missing."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            if missing_ok:
                return ""
            raise
        except Exception as e:
            print(f"[error] Failed to read {path}: {e}")
            raise

    @staticmethod
    def _to_text(value: Any, max_len: Optional[int] = None) -> str:
        """Normalize arbitrary values into a trimmed string (safe for prompts)."""
        try:
            if isinstance(value, str):
                s = value
            elif isinstance(value, (dict, list)):
                s = json.dumps(value, ensure_ascii=False)
            elif value is None:
                s = ""
            else:
                s = str(value)
        except Exception:
            s = ""
        s = s.strip()
        if max_len and len(s) > max_len:
            s = s[:max_len]
        return s

    @staticmethod
    def _slugify(name: str) -> str:
        """Simple slug for filenames."""
        return (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace(",", "")
            .replace(":", "")
        )
