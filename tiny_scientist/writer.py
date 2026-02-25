import json
import os
import os.path as osp
import re
import subprocess
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, Runner
from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .smolagents_tools import DrawerTool, PaperSearchTool
from .tools.agent_tools import build_research_tools
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
from .utils.sdk_client import configure_openai_agents_for_model, track_sdk_cost

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


class _WriterLegacy:
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
        self.searcher: PaperSearchTool = PaperSearchTool(
            s2_api_key=s2_api_key,
            engine="semanticscholar",
        )
        self.drawer: DrawerTool = DrawerTool(model, prompt_template_dir, temperature)

        # Prompts & config (load first)
        self.config = Config(prompt_template_dir)
        self.prompts = self.config.prompt_template.writer_prompt

        # Formatter selection (with latex_fix_prompt)
        if self.template == "acl":
            self.formatter: BaseOutputFormatter = ACLOutputFormatter(
                model=self.model,
                client=self.client,
                latex_fix_prompt=self.config.prompt_template.latex_fix_prompt,
            )
        elif self.template == "iclr":
            self.formatter = ICLROutputFormatter(
                model=self.model,
                client=self.client,
                latex_fix_prompt=self.config.prompt_template.latex_fix_prompt,
            )
        else:
            raise ValueError(f"Unknown template: {self.template!r}")
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
        self._write_reference_links(idea)
        self.cost_tracker.report()
        return output_pdf_path, paper_name

    # ---- Section generation -------------------------------------------------

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        """Generate abstract using all already-written sections as context."""
        title = idea.get("Title", "Research Paper")
        full_context = self._sections_as_markdown(exclude={"Abstract"})
        # Use unified idea string in abstract prompt as requested
        idea_str = self._format_idea_as_string(idea)
        abstract_prompt = self.prompts.abstract_prompt.format(
            abstract_tips=self.prompts.section_tips.get("Abstract", ""),
            idea=idea_str,
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
        """Ask LLM to propose citation search queries."""
        # Build a single idea string for searching
        idea_str = self._format_idea_as_string(idea)

        prompt = self.prompts.citation_search_query_prompt.format(
            idea=idea_str,
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
            link = str(pdata.get("url") or pdata.get("source") or "").strip()
            bibtex = pdata.get("bibtex", "") or ""

            bibkey = "UNKNOWN"
            m = re.search(r"@\w+\{(.+?),", bibtex)
            if m:
                bibkey = m.group(1).strip()

            # Use format that doesn't conflict with .format() syntax
            entry = f"- [CITE_KEY: {bibkey}] {title} ({authors}, {venue}, {year})\n  Abstract: {abstract}"
            if link:
                entry += f"\n  URL: {link}"
            entries.append(entry)

        return "\n\n".join(entries)

    def _collect_reference_links(self, idea: Dict[str, Any]) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        seen: set[str] = set()

        def _add(title: str, url: str, source_type: str) -> None:
            clean_url = (url or "").strip()
            if not clean_url.startswith(("http://", "https://")):
                return
            key = clean_url.lower()
            if key in seen:
                return
            seen.add(key)
            links.append(
                {
                    "title": (title or "Untitled").strip() or "Untitled",
                    "url": clean_url,
                    "source_type": (source_type or "unknown").strip() or "unknown",
                }
            )

        for title, pdata in self.references.items():
            if not isinstance(pdata, dict):
                continue
            _add(
                title=str(title),
                url=str(pdata.get("url") or pdata.get("source") or ""),
                source_type=str(pdata.get("source_type") or "paper_search"),
            )

        citations = idea.get("Citations", [])
        if isinstance(citations, list):
            for c in citations:
                if not isinstance(c, dict):
                    continue
                _add(
                    title=str(c.get("title", "Untitled")),
                    url=str(c.get("url", "")),
                    source_type=str(c.get("source_type", "idea_citation")),
                )

        grounding = idea.get("ResearchGrounding", {})
        if isinstance(grounding, dict):
            grounded_citations = grounding.get("citations", [])
            if isinstance(grounded_citations, list):
                for c in grounded_citations:
                    if not isinstance(c, dict):
                        continue
                    _add(
                        title=str(c.get("title", "Untitled")),
                        url=str(c.get("url", "")),
                        source_type=str(c.get("source_type", "research_grounding")),
                    )

        return links

    def _write_reference_links(self, idea: Dict[str, Any]) -> None:
        links = self._collect_reference_links(idea)
        path = osp.join(self.output_dir, "reference_links.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reference_links": links}, f, indent=2, ensure_ascii=False)

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


class Writer(_WriterLegacy):
    """Writer variant that uses the OpenAI Agents SDK for all LLM calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        configure_openai_agents_for_model(self.model)
        shared_tools = build_research_tools(model=self.model, include_drawer=True)
        tool_policy = (
            "Tool policy: use web_search for recent context, paper_search for academic "
            "evidence, code_search for reproducibility references, and generate_diagram "
            "for method/result visuals."
        )

        self.write_agent = Agent(
            name="PaperWriter",
            instructions=f"{self.system_prompt}\n\n{tool_policy}",
            tools=shared_tools,
            model=self.model,
        )
        self._related_work_agent = Agent(
            name="RelatedWorkWriter",
            instructions=f"{self.prompts.write_system_prompt_related_work}\n\n{tool_policy}",
            tools=shared_tools,
            model=self.model,
        )
        self.citation_agent = Agent(
            name="CitationEnricher",
            instructions=f"{self.prompts.citation_system_prompt}\n\n{tool_policy}",
            tools=shared_tools,
            model=self.model,
        )
        self.visual_planner_agent = Agent(
            name="VisualPlanner",
            instructions=(
                "You are a scientific visualization planner. "
                "Return compact JSON with keys 'figures' and 'tables'. "
                "Each figure item: {name, section, goal}. "
                "Each table item: {name, goal}. "
                "Prefer <=2 figures and <=1 table."
            ),
            tools=shared_tools,
            model=self.model,
        )
        self.table_agent = Agent(
            name="TableComposer",
            instructions=(
                "You write strict LaTeX tables for research papers. "
                "Return only LaTeX for one complete table environment with caption and label."
            ),
            tools=shared_tools,
            model=self.model,
        )
        self.latex_error_agent = Agent(
            name="LaTeXErrorHandler",
            instructions=(
                "You fix LaTeX compilation failures for academic papers. "
                "Use web_search to verify package/command usage when needed. "
                "Return only the full corrected .tex content, no markdown fences."
            ),
            tools=shared_tools,
            model=self.model,
        )
        self.bib_manager_agent = Agent(
            name="BibManager",
            instructions=(
                "You are a bibliography manager for ML papers. "
                "Normalize and deduplicate references, ensure each entry has a valid citation key and bibtex. "
                "Use paper_search/web_search to fill missing metadata when needed. "
                "Return ONLY JSON with key 'references' as a list. "
                "Each item must include: title, authors, venue, year, abstract, url, source_type, citation_key, bibtex. "
                "If url is present it must be http/https."
            ),
            tools=shared_tools,
            model=self.model,
        )
        self.planner_agent = Agent(
            name="WriterPlanner",
            instructions=(
                "You are a paper-writing execution planner. "
                "Return only a JSON array of TODO items. "
                "Each item must include {step, action, name, description}. "
                "Allowed actions: write_related_work, write_section, write_visuals, "
                "write_abstract, refine_paper, format_export. "
                "For write_section include a 'section' field."
            ),
            tools=shared_tools,
            model=self.model,
        )

    def _run_sdk_call(self, agent: Agent, prompt: str, task_name: str) -> str:
        result = Runner.run_sync(agent, prompt)
        track_sdk_cost(result, self.cost_tracker, self.model, task_name)
        return result.final_output or ""

    def run(
        self, idea: Dict[str, Any], experiment_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        """SDK writer pipeline with planner TODO execution."""
        is_experimental = idea.get("is_experimental", True)
        code, experiment_result, baseline_result = self._load_experiment_bundle(
            is_experimental, experiment_dir
        )
        sections = (
            REFINEMENT_SECTIONS_EXPERIMENTAL
            if is_experimental
            else REFINEMENT_SECTIONS_NONEXP
        )
        todo = self._build_todo(idea=idea, sections=sections)
        print(f"[Planner][Writer] TODO ({len(todo)} steps)")
        for item in todo:
            print(
                f"[TODO][Writer] [ ] Step {item['step']}: {item['name']} — {item['description']}"
            )

        paper_name = self._slugify(idea.get("Title", "Research Paper"))
        output_pdf_path = f"{self.output_dir}/{paper_name}.pdf"
        formatted = False
        for idx, item in enumerate(todo, start=1):
            action = str(item.get("action", ""))
            print(f"[TODO][Writer] [{idx}/{len(todo)}] {item.get('name', action)}")
            if action == "write_related_work":
                self._write_related_work(idea)
            elif action == "write_section":
                section = str(item.get("section", ""))
                if section in sections:
                    self._write_section(
                        idea, code, experiment_result, section, baseline_result
                    )
            elif action == "write_visuals":
                self._run_visual_subagents(
                    idea=idea,
                    experiment_result=experiment_result,
                    baseline_result=baseline_result,
                )
            elif action == "write_abstract":
                self._write_abstract(idea)
            elif action == "refine_paper":
                self._refine_paper(
                    num_rounds=self.num_refinement_rounds, add_citations=True
                )
            elif action == "format_export":
                self.references = self._manage_bibliography_references(idea=idea)
                self.formatter.run(
                    content=self.generated_sections,
                    references=self.references,
                    output_dir=self.output_dir,
                    output_pdf_path=output_pdf_path,
                    name=self.generated_sections.get("Title", "Research Paper"),
                )
                formatted = True
        if not formatted:
            self.references = self._manage_bibliography_references(idea=idea)
            self.formatter.run(
                content=self.generated_sections,
                references=self.references,
                output_dir=self.output_dir,
                output_pdf_path=output_pdf_path,
                name=self.generated_sections.get("Title", "Research Paper"),
            )
        self._recover_latex_if_missing_pdf(output_pdf_path)
        self._write_reference_links(idea)
        self.cost_tracker.report()
        return output_pdf_path, paper_name

    def _manage_bibliography_references(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        raw_entries: List[Dict[str, Any]] = []

        for title, pdata in self.references.items():
            if not isinstance(pdata, dict):
                continue
            raw_entries.append(
                {
                    "title": str(title),
                    "authors": pdata.get("authors", ""),
                    "venue": pdata.get("venue", ""),
                    "year": pdata.get("year", ""),
                    "abstract": pdata.get("abstract", ""),
                    "url": pdata.get("url", pdata.get("source", "")),
                    "source_type": pdata.get("source_type", "paper_search"),
                    "bibtex": pdata.get("bibtex", ""),
                }
            )

        for c in idea.get("Citations", []) if isinstance(idea.get("Citations", []), list) else []:
            if not isinstance(c, dict):
                continue
            raw_entries.append(
                {
                    "title": str(c.get("title", "Untitled")),
                    "authors": "",
                    "venue": "",
                    "year": "",
                    "abstract": "",
                    "url": str(c.get("url", "")),
                    "source_type": str(c.get("source_type", "idea_citation")),
                    "bibtex": "",
                }
            )

        prompt = (
            "Manage bibliography for this paper. Deduplicate and complete entries.\n\n"
            f"Paper title: {idea.get('Title', '')}\n"
            "Current references JSON:\n"
            f"{json.dumps(raw_entries, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only."
        )
        text = self._run_sdk_call(self.bib_manager_agent, prompt, "manage_bibliography")
        parsed = extract_json_between_markers(text)
        if not isinstance(parsed, dict):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        if not isinstance(parsed, dict):
            raise RuntimeError("[Writer][BibManager] invalid JSON output.")

        refs = parsed.get("references", [])
        if not isinstance(refs, list) or not refs:
            raise RuntimeError("[Writer][BibManager] no references returned.")

        normalized: Dict[str, Any] = {}
        manifest: List[Dict[str, Any]] = []
        for entry in refs:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title", "")).strip()
            bibtex = str(entry.get("bibtex", "")).strip()
            if not title or not bibtex:
                continue
            url = str(entry.get("url", "")).strip()
            if url and not url.startswith(("http://", "https://")):
                continue
            citation_key = str(entry.get("citation_key", "")).strip()
            normalized[title] = {
                "title": title,
                "authors": entry.get("authors", ""),
                "venue": entry.get("venue", ""),
                "year": entry.get("year", ""),
                "abstract": entry.get("abstract", ""),
                "url": url,
                "source_type": entry.get("source_type", "bib_manager"),
                "citation_key": citation_key,
                "bibtex": bibtex,
            }
            manifest.append(normalized[title])

        if not normalized:
            raise RuntimeError("[Writer][BibManager] all references invalid after normalization.")

        manifest_path = osp.join(self.output_dir, "bibliography_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"references": manifest}, f, indent=2, ensure_ascii=False)
        print(f"[Writer][BibManager] managed {len(normalized)} references -> {manifest_path}")
        return normalized

    def _recover_latex_if_missing_pdf(self, output_pdf_path: str) -> None:
        if osp.exists(output_pdf_path):
            return
        latex_dir = osp.join(self.output_dir, "latex")
        tex_file = (
            "acl_latex.tex" if self.template == "acl" else "iclr2025_conference.tex"
        )
        tex_path = osp.join(latex_dir, tex_file)
        log_path = osp.join(latex_dir, tex_file.replace(".tex", ".log"))
        if not osp.exists(tex_path):
            raise RuntimeError(
                f"LaTeX compilation failed and tex source not found: {tex_path}"
            )

        for attempt in range(2):
            tex_content = self._read_text(tex_path)
            log_excerpt = self._read_text(log_path, missing_ok=True)[-12000:]
            prompt = (
                "LaTeX compilation failed. Produce a corrected full .tex file.\n\n"
                f"Template: {self.template}\n"
                f"Attempt: {attempt + 1}\n\n"
                "Compiler log excerpt:\n"
                f"{log_excerpt}\n\n"
                "Current tex:\n"
                f"{tex_content}"
            )
            candidate = self._run_sdk_call(
                self.latex_error_agent, prompt, "latex_error_recovery"
            )
            fixed_tex = self._extract_tex(candidate)
            if not fixed_tex.strip():
                raise RuntimeError(
                    "[Writer][LaTeX] latex_error_agent returned empty tex content."
                )
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(fixed_tex)

            self._compile_latex_once(latex_dir, tex_file)
            produced_pdf = osp.join(latex_dir, tex_file.replace(".tex", ".pdf"))
            if osp.exists(produced_pdf):
                os.makedirs(osp.dirname(output_pdf_path), exist_ok=True)
                if osp.exists(output_pdf_path):
                    os.remove(output_pdf_path)
                os.replace(produced_pdf, output_pdf_path)
                return

        raise RuntimeError(
            "[Writer][LaTeX] failed to compile PDF after latex_error_agent recovery attempts."
        )

    @staticmethod
    def _extract_tex(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    def _compile_latex_once(self, cwd: str, tex_file: str) -> None:
        base = tex_file.replace(".tex", "")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-file-line-error", tex_file],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        subprocess.run(
            ["bibtex", base],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-file-line-error", tex_file],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-file-line-error", tex_file],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )

    def _build_todo(self, idea: Dict[str, Any], sections: List[str]) -> List[Dict[str, Any]]:
        prompt = (
            "Build writing TODO for this idea.\n"
            f"Idea:\n{self._format_idea_as_string(idea)[:3000]}\n\n"
            f"Required sections: {sections}\n"
            "Return JSON array only."
        )
        text = self._run_sdk_call(self.planner_agent, prompt, "plan_writer_todo")
        parsed = extract_json_between_markers(text)
        if not isinstance(parsed, list):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        if isinstance(parsed, list) and parsed:
            allowed = {
                "write_related_work",
                "write_section",
                "write_visuals",
                "write_abstract",
                "refine_paper",
                "format_export",
            }
            normalized: List[Dict[str, Any]] = []
            for idx, item in enumerate(parsed, start=1):
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action", "")).strip()
                if action not in allowed:
                    continue
                section = str(item.get("section", "")).strip()
                if action == "write_section" and section not in sections:
                    continue
                normalized.append(
                    {
                        "step": int(item.get("step", idx)),
                        "action": action,
                        "name": str(item.get("name", action)),
                        "description": str(item.get("description", "")),
                        **({"section": section} if section else {}),
                    }
                )
            actions = {str(it.get("action", "")) for it in normalized}
            if (
                normalized
                and "write_related_work" in actions
                and "write_abstract" in actions
                and "format_export" in actions
            ):
                return normalized
        raise RuntimeError(
            "[Planner][Writer] invalid TODO from planner: must include related_work, abstract, and export steps."
        )

    def _run_visual_subagents(
        self, idea: Dict[str, Any], experiment_result: str, baseline_result: str
    ) -> None:
        """Plan and materialize figures/tables via dedicated sub-agents."""
        plan_prompt = (
            "Research idea:\n"
            f"{self._format_idea_as_string(idea)}\n\n"
            "Current section status:\n"
            f"{self._sections_as_markdown(exclude={'Abstract'})[:5000]}\n\n"
            "Experiment results:\n"
            f"{experiment_result[:3000]}\n\n"
            "Baseline results:\n"
            f"{baseline_result[:2000]}"
        )
        plan_text = self._run_sdk_call(
            self.visual_planner_agent, plan_prompt, "plan_visual_assets"
        )
        plan = extract_json_between_markers(plan_text)
        if not isinstance(plan, dict):
            try:
                plan = json.loads(plan_text)
            except Exception:
                plan = {}

        figures = plan.get("figures", []) if isinstance(plan, dict) else []
        tables = plan.get("tables", []) if isinstance(plan, dict) else []

        self._materialize_figure_assets(figures)
        self._materialize_results_table(tables, experiment_result, baseline_result)

    def _materialize_figure_assets(self, figures: Any) -> None:
        if not isinstance(figures, list) or not figures:
            return
        assets_dir = osp.join(self.output_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        manifest: List[Dict[str, str]] = []

        for idx, fig in enumerate(figures[:2], start=1):
            section = str(fig.get("section", "Results")) if isinstance(fig, dict) else "Results"
            name = str(fig.get("name", f"figure_{idx}")) if isinstance(fig, dict) else f"figure_{idx}"
            goal = str(fig.get("goal", "")) if isinstance(fig, dict) else ""
            section_content = self.generated_sections.get(section, "")[:4000]
            if not section_content:
                continue
            result = self.drawer.run(
                json.dumps(
                    {
                        "section_name": section if section in self.prompts.section_prompt else "Results",
                        "section_content": section_content,
                    }
                )
            )
            diagram = result.get("diagram", {})
            svg = diagram.get("svg", "")
            if not svg:
                continue
            filename = f"{self._slugify(name)}.svg"
            path = osp.join(assets_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
            manifest.append(
                {
                    "name": name,
                    "section": section,
                    "goal": goal,
                    "summary": diagram.get("summary", ""),
                    "path": path,
                }
            )

        if manifest:
            manifest_path = osp.join(assets_dir, "figure_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(f"[visual] Saved figure manifest: {manifest_path}")

    def _materialize_results_table(
        self, tables: Any, experiment_result: str, baseline_result: str
    ) -> None:
        if not isinstance(tables, list) or not tables:
            return
        table_goal = ""
        first = tables[0]
        if isinstance(first, dict):
            table_goal = str(first.get("goal", "Summarize key metrics and comparisons"))
        prompt = (
            "Create one high-signal Results table in LaTeX.\n"
            f"Goal: {table_goal}\n\n"
            "Experiment results:\n"
            f"{experiment_result[:5000]}\n\n"
            "Baseline results:\n"
            f"{baseline_result[:3000]}"
        )
        table_latex = self._run_sdk_call(self.table_agent, prompt, "compose_results_table")
        table_latex = table_latex.strip()
        if "\\begin{table" not in table_latex or "\\end{table" not in table_latex:
            return
        current_results = self.generated_sections.get("Results", "")
        if "\\begin{table" in current_results:
            return
        self.generated_sections["Results"] = f"{current_results}\n\n{table_latex}"

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        full_context = self._sections_as_markdown(exclude={"Abstract"})
        idea_str = self._format_idea_as_string(idea)
        abstract_prompt = self.prompts.abstract_prompt.format(
            abstract_tips=self.prompts.section_tips.get("Abstract", ""),
            idea=idea_str,
            full_paper_content=full_context,
        )
        abstract_content = self._run_sdk_call(
            self.write_agent, abstract_prompt, "Abstract"
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
        ctx = self._build_section_context(idea, code, experiment_result, baseline_result)
        ctx["section_tips"] = self.prompts.section_tips.get(section, "")

        template = self.prompts.section_prompt.get(section)
        if not template:
            print(f"[warn] No prompt template for section: {section}")
            return

        prompt = template.format(**ctx)
        content = self._run_sdk_call(self.write_agent, prompt, f"{section} section")
        self.generated_sections[section] = content

        try:
            enriched = self._enrich_section_with_citations(
                section=section, idea=idea, max_queries=5
            )
            if enriched:
                self.generated_sections[section] = enriched
        except Exception as e:
            print(f"[warn] Citation enrichment failed in {section}: {e}")
            traceback.print_exc()

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        try:
            queries = self._generate_search_queries(
                idea, section="Related_Work", max_queries=6
            )
            papers = self._search_papers_by_queries(queries)
            if not papers:
                print("[warn] No papers found for Related Work")
                return

            self.references = papers

            paper_ctx = self._format_paper_context(papers)
            experiment = idea.get("Experiment", "No experiment details provided")
            prompt = self.prompts.related_work_prompt.format(
                related_work_tips=self.prompts.section_tips["Related_Work"],
                experiment=experiment,
                references=paper_ctx,
            )
            content = self._run_sdk_call(
                self._related_work_agent, prompt, "Related_Work"
            )
            self.generated_sections["Related_Work"] = content
            print(f"[info] Related Work generated with {len(papers)} references")
        except Exception as e:
            print(f"[warn] Related Work generation failed: {e}")
            traceback.print_exc()

    def _enrich_section_with_citations(
        self,
        section: str,
        idea: Dict[str, Any],
        max_queries: int = 4,
    ) -> Optional[str]:
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

        for title, pdata in papers.items():
            self.references.setdefault(title, pdata)

        paper_ctx = self._format_paper_context(papers)
        embed_prompt = self.prompts.add_new_citations_prompt.format(
            section=section,
            section_content=original,
            paper_context=paper_ctx,
            num_papers=len(papers),
        )
        enriched = self._run_sdk_call(
            self.citation_agent, embed_prompt, f"Add_Citations_{section}"
        )
        return enriched

    def _refine_paper(self, num_rounds: int = 2, add_citations: bool = True) -> None:
        remaining_budget = self._effective_remaining_budget()

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

        refined_title = self._run_sdk_call(
            self.write_agent, title_prompt, "Title Refinement"
        )
        self.generated_sections["Title"] = refined_title

        refinement_priority = [
            "Method",
            "Introduction",
            "Experimental_Setup",
            "Results",
            "Discussion",
            "Conclusion",
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
                            f"[Writer] Skipping refinement for {section} in round {r} "
                            "due to estimated budget constraints."
                        )
                        continue

                refined = self._run_sdk_call(
                    self.write_agent, prompt, f"Refine_R{r}_{section}"
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

    def _generate_search_queries(
        self,
        idea: Dict[str, Any],
        section: str = "",
        content_snippet: str = "",
        max_queries: int = 6,
    ) -> List[str]:
        idea_str = self._format_idea_as_string(idea)
        prompt = self.prompts.citation_search_query_prompt.format(
            idea=idea_str,
            section=section or "General",
            snippet=content_snippet or "",
        )
        resp = self._run_sdk_call(
            self.citation_agent,
            prompt,
            f"generate_queries_{section or 'general'}",
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
