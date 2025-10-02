import json
import os
import os.path as osp
import re
import time
import traceback
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple

import cairosvg
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


class Writer:
    def __init__(
        self,
        model: str,
        output_dir: str,
        template: str,
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
        cost_tracker: Optional[BudgetChecker] = None,
        s2_api_key: Optional[str] = None,
        num_refinement_rounds: int = 2,  # Number of refinement rounds
    ) -> None:
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.template = template
        self.temperature = temperature
        self.num_refinement_rounds = num_refinement_rounds
        # If not provided explicitly, read S2_API_KEY from environment
        s2_api_key = s2_api_key or os.environ.get("S2_API_KEY")
        # Use Semantic Scholar only, no fallback to OpenAlex
        # Note: If you want to enable OpenAlex fallback, set disable_fallback=False
        self.searcher: BaseTool = PaperSearchTool(
            s2_api_key=s2_api_key, 
            engine="semanticscholar",
            disable_fallback=True  # Disable OpenAlex fallback
        )
        self.drawer: BaseTool = DrawerTool(model, prompt_template_dir, temperature)
        self.formatter: BaseOutputFormatter
        self.config = Config(prompt_template_dir)
        if self.template == "acl":
            self.formatter = ACLOutputFormatter(model=self.model, client=self.client)
        elif self.template == "iclr":
            self.formatter = ICLROutputFormatter(model=self.model, client=self.client)

        self.prompts = self.config.prompt_template.writer_prompt
        self.cost_tracker = cost_tracker or BudgetChecker()

        with resources.files("tiny_scientist.fewshot_sample").joinpath(
            "automated_relational.txt"
        ).open("r", encoding="utf-8") as f:
            few_shot_sample_text = f.read()

        self.system_prompt = self.prompts.write_system_prompt.format(
            example_paper_draft=few_shot_sample_text
        )

    def run(
        self, idea: Dict[str, Any], experiment_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        is_experimental = idea.get("is_experimental", True)

        code, experiment_result, baseline_result = "", "", ""

        if is_experimental and experiment_dir:
            # Load experiment files for experimental papers
            with open(osp.join(experiment_dir, "experiment.py"), "r") as f:
                code = f.read()

            with open(osp.join(experiment_dir, "experiment_results.txt"), "r") as f:
                experiment_result = f.read()

            if osp.exists(osp.join(experiment_dir, "baseline_results.txt")):
                with open(osp.join(experiment_dir, "baseline_results.txt"), "r") as f:
                    baseline_result = f.read()
        elif is_experimental and not experiment_dir:
            raise ValueError("Experimental papers require an experiment_dir")

        self.generated_sections: Dict[str, Any] = {}
        self.references: Dict[str, Any] = {}

        # First generate Related Work to provide background context
        print("Generating Related Work section...")
        try:
            self._write_related_work(idea)
        except Exception as e:
            print(f"[WARNING] Failed to generate Related Work: {e}")
            # Continue without Related Work if it fails
            pass

        # Different section structures for experimental vs non-experimental papers
        if is_experimental:
            sections = [
                "Introduction",
                "Method",
                "Experimental_Setup",
                "Results",
                "Discussion",
                "Conclusion",
            ]
        else:
            sections = [
                "Introduction",
                "Method",
                "Analysis",
                "Discussion",
                "Conclusion",
            ]

        # Generate all main sections
        for section in sections:
            self._write_section(idea, code, experiment_result, section, baseline_result)

        # Generate Abstract last, after all sections are complete
        print("Generating Abstract (final summary)...")
        self._write_abstract(idea)

        # Multi-round refinement with progressive citation enrichment
        self._refine_paper(num_rounds=self.num_refinement_rounds, add_citations=True)

        #self._generate_diagram_for_section()

        paper_name = (
            idea.get("Title", "Research Paper")
            .lower()
            .replace(" ", "_")
            .lower()
            .replace(" ", "_")
        )

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

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        """Generate abstract based on all completed sections."""
        title = idea.get("Title", "Research Paper")
        
        # Build full paper context from all generated sections
        full_paper_context = ""
        if self.generated_sections:
            context_parts = []
            for section, content in self.generated_sections.items():
                if section not in ["Abstract"]:  # Skip abstract itself
                    context_parts.append(f"## {section}\n{content}")
            if context_parts:
                full_paper_context = "\n\n".join(context_parts)

        abstract_prompt = self.prompts.abstract_prompt.format(
            abstract_tips=self.prompts.section_tips.get("Abstract", ""),
            title=title,
            problem=idea.get("Problem", ""),
            importance=idea.get("Importance", ""),
            difficulty=idea.get("Difficulty", ""),
            novelty=idea.get("NoveltyComparison", ""),
            experiment=idea.get("Experiment", ""),
            full_paper_content=full_paper_context,  # Add full paper context
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

    def _generate_diagram_for_section(self) -> None:
        for section in ["Method", "Experimental_Setup", "Results"]:
            content = self.generated_sections[section]
            try:
                query = json.dumps(
                    {"section_name": section, "section_content": content}
                )
                diagram_result = self.drawer.run(query)

                if diagram_result and "diagram" in diagram_result:
                    diagram = diagram_result["diagram"]

                    pdf_filename = f"diagram_{section.lower()}.pdf"
                    pdf_path = os.path.join(self.output_dir, "latex", pdf_filename)
                    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

                    raw_svg = diagram["svg"]

                    cleaned_svg = raw_svg.encode("utf-8").decode("unicode_escape")
                    cleaned_svg = cleaned_svg.replace("\\n", "\n").replace("\\", "")

                    raw_svg = diagram["svg"]

                    cleaned_svg = raw_svg.encode("utf-8").decode("unicode_escape")
                    cleaned_svg = cleaned_svg.replace("\\n", "\n").replace("\\", "")

                    cairosvg.svg2pdf(
                        bytestring=cleaned_svg.encode("utf-8"), write_to=pdf_path
                    )

                    # Sanitize caption to avoid LaTeX errors
                    caption = diagram["summary"].replace("{", "").replace("}", "")

                    figure_latex = f"""
            \\begin{{figure}}[!htbp]
            \\centering
            \\includegraphics[width=0.9\\linewidth]{{{pdf_filename}}}
            \\caption{{{caption}}}
            \\label{{fig:{section.lower()}}}
            \\end{{figure}}
            """
                    marker = "```"

                    if self.generated_sections[section].strip().endswith(marker):
                        parts = content.strip().rsplit(marker, 1)
                        if len(parts) == 2:
                            self.generated_sections[section] = (
                                parts[0].strip()
                                + "\n"
                                + figure_latex
                                + "\n"
                                + marker
                                + parts[1]
                            )

            except Exception as e:
                print(f"[WARNING] Failed to generate diagram for {section}: {e}")
                traceback.print_exc()

        return None

    def _extract_method_details_from_code(self, code: str) -> str:
        """Extract method-relevant details from experiment code."""
        if not code:
            return ""
        
        details = []
        
        # Extract class definitions (potential model architectures)
        class_matches = re.findall(r'class\s+(\w+)\s*\([^)]*\):', code)
        if class_matches:
            details.append(f"Model classes: {', '.join(class_matches[:5])}")
        
        # Extract key hyperparameters
        hyperparam_patterns = [
            r'learning_rate\s*=\s*([0-9.e-]+)',
            r'batch_size\s*=\s*(\d+)',
            r'hidden_size\s*=\s*(\d+)',
            r'num_layers\s*=\s*(\d+)',
            r'dropout\s*=\s*([0-9.]+)',
            r'epochs?\s*=\s*(\d+)',
        ]
        for pattern in hyperparam_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                param_name = pattern.split(r'\s*=')[0].replace(r'\\s*', '').strip(r'\\')
                details.append(f"{param_name}: {matches[0]}")
        
        # Extract optimizer info
        optimizer_match = re.search(r'(\w*Adam\w*|SGD|RMSprop)', code)
        if optimizer_match:
            details.append(f"Optimizer: {optimizer_match.group(1)}")
        
        return "\n".join(details) if details else ""
    
    def _build_section_context(self, idea: Dict[str, Any], code: str, 
                                experiment_result: str, baseline_result: str) -> Dict[str, Any]:
        """Build all possible context variables for section prompts."""
        # Build previous context from all generated sections (except Abstract)
        previous_context = ""
        if self.generated_sections:
            context_parts = []
            for prev_section, content in self.generated_sections.items():
                if prev_section not in ["Abstract"]:
                    context_parts.append(f"## {prev_section}\n{content}")
            if context_parts:
                previous_context = "\n\n".join(context_parts)

        # Extract method details from code
        code_method_details = self._extract_method_details_from_code(code)

        # Prepare all possible variables that any section might need
        return {
            "title": idea.get("Title", "Research Paper"),
            "problem": idea.get("Problem", ""),
            "importance": idea.get("Importance", ""),
            "difficulty": idea.get("Difficulty", ""),
            "novelty": idea.get("NoveltyComparison", ""),
            "approach": idea.get("Approach", ""),  # IMPORTANT: Proposed approach from thinker
            "experiment": idea.get("Experiment", idea.get("ResearchPlan", "")),
            "code": code,
            "code_method_details": code_method_details,  # NEW: Extracted method details
            "experiment_results": experiment_result,
            "baseline_results": baseline_result,
            "previous_context": previous_context,
            # Abstract will be empty during main section generation (it's generated last)
            "abstract_content": self.generated_sections.get("Abstract", ""),
            "related_work_content": self.generated_sections.get("Related_Work", ""),
        }

    def _write_section(
        self,
        idea: Dict[str, Any],
        code: str,
        experiment_result: str,
        section: str,
        baseline_result: Optional[str] = "",
    ) -> None:
        print(f"Writing section: {section}...")

        # Build all context variables
        context = self._build_section_context(idea, code, experiment_result, baseline_result or "")

        # Add section-specific tips
        context["section_tips"] = self.prompts.section_tips.get(section, "")

        # Get section prompt template and format with all context
        section_prompt_template = self.prompts.section_prompt.get(section)
        if not section_prompt_template:
            print(f"[WARNING] No prompt template for section: {section}")
            return

        # Format prompt with all available context (template will use what it needs)
        section_prompt = section_prompt_template.format(**context)

        # Generate section content
        section_content, _ = get_response_from_llm(
            msg=section_prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"{section} section",
        )

        self.generated_sections[section] = section_content

        # Enrich section with citations
        try:
            enriched = self._enrich_section_with_citations(section, idea)
            if enriched:
                self.generated_sections[section] = enriched
        except Exception as e:
            print(f"[WARNING] Failed to enrich citations for section {section}: {e}")
            traceback.print_exc()

    def _convert_to_text(self, value: Any, max_len: Optional[int] = None) -> str:
        """Convert arbitrary value (dict/list/None/str/other) to a clean string.

        - dict/list -> JSON string
        - None -> ""
        - truncate to max_len if provided
        """
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
        if max_len is not None and len(s) > max_len:
            s = s[:max_len]
        return s

    def _generate_search_queries(
        self, idea: Dict[str, Any], section: str = "", content_snippet: str = "", max_queries: int = 6
    ) -> List[str]:
        """Generate search queries using LLM from idea/section content."""
        title = self._convert_to_text(idea.get("Title", ""))
        problem = self._convert_to_text(idea.get("Problem", ""))
        novelty = self._convert_to_text(idea.get("NoveltyComparison", ""))
        experiment = self._convert_to_text(idea.get("Experiment", ""))

        prompt = self.prompts.citation_search_query_prompt.format(
            idea_title=title or "Research Paper",
            problem=problem or "",
            novelty=novelty or "",
            experiment=experiment or "",
            section=section or "General",
            snippet=content_snippet or "",
        )

        response, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.citation_system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"generate_queries_{section or 'general'}",
        )

        queries: List[str] = []
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                queries = [self._convert_to_text(q) for q in parsed if self._convert_to_text(q)]
        except json.JSONDecodeError:
            parsed = extract_json_between_markers(response)
            if isinstance(parsed, list):
                queries = [self._convert_to_text(q) for q in parsed if self._convert_to_text(q)]

        return queries[:max_queries]

    def _search_papers_by_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Search papers by multiple queries and merge results (abstract + bibtex)."""
        merged_papers: Dict[str, Any] = {}
        for query in queries:
            try:
                results = self.searcher.run(query)
                if results:
                    for title, paper_data in results.items():
                        if title not in merged_papers:
                            merged_papers[title] = paper_data
                time.sleep(0.8)
            except Exception as e:
                print(f"[ERROR] Search failed for query '{query}': {e}")
                traceback.print_exc()
        return merged_papers

    def _format_paper_context(self, papers: Dict[str, Any]) -> str:
        """Format papers into context string with abstracts AND bibtex keys for LLM prompt."""
        paper_entries = []
        for title, paper_data in papers.items():
            authors = self._format_authors(paper_data.get("authors", ""))
            abstract = paper_data.get("abstract", "")[:400]  # Truncate long abstracts
            year = paper_data.get("year", "")
            venue = paper_data.get("venue", "")
            
            # Extract bibtex key
            bibtex = paper_data.get("bibtex", "")
            bibtex_key = "UNKNOWN"
            if bibtex:
                match = re.search(r"@\w+\{(.+?),", bibtex)
                if match:
                    bibtex_key = match.group(1).strip()
            
            # Include bibtex key in the context so LLM uses correct citation
            entry = f"- **[{bibtex_key}]** {title} ({authors}, {venue}, {year})\n  Abstract: {abstract}"
            paper_entries.append(entry)
        
        context = "\n\n".join(paper_entries)
        # Escape for format strings
        return context.replace("{", "{{").replace("}", "}}")

    def _replace_titles_with_bibtex_keys(self, content: str, papers: Dict[str, Any]) -> str:
        """Replace \\cite{Paper Title} with \\cite{bibtex_key} in content."""
        updated = content
        for title, paper_data in papers.items():
            bibtex = paper_data.get("bibtex", "")
            match = re.search(r"@\w+\{(.+?),", bibtex)
            if match:
                bibtex_key = match.group(1)
                # Use literal string matching instead of regex to avoid escape issues
                cite_pattern = f"\\cite{{{title}}}"
                # Also try with whitespace variations
                patterns_to_try = [
                    f"\\cite{{{title}}}",
                    f"\\cite{{ {title} }}",
                    f"\\cite{{ {title}}}",
                    f"\\cite{{{title} }}",
                ]
                
                for pattern in patterns_to_try:
                    if pattern in updated:
                        updated = updated.replace(pattern, f"\\cite{{{bibtex_key}}}")
                        break
        return updated

    def _enrich_section_with_citations_v2(
        self, 
        section: str, 
        max_queries: int = 3,
        max_papers_per_query: int = 2
    ) -> Optional[str]:
        """
        Enhanced citation enrichment with control over search volume.
        Avoids re-searching for papers we already have.
        """
        original_content = self.generated_sections.get(section, "")
        if not original_content or len(original_content) < 50:
            return None

        # Extract existing citations to avoid duplication
        existing_citations = set(re.findall(r'\\cite\{([^\}]+)\}', original_content))
        existing_keys = set()
        for cite in existing_citations:
            existing_keys.update([k.strip() for k in cite.split(',')])
        
        print(f"[Citation] Section {section} already has {len(existing_keys)} citations")

        # Generate focused queries based on section content
        idea_proxy = {
            "Title": self.generated_sections.get("Title", ""),
            "Problem": section,  # Use section name as focus
            "Experiment": original_content[:300],  # Use section content
        }
        
        queries = self._generate_search_queries(
            idea_proxy, 
            section=section, 
            content_snippet=original_content[:600], 
            max_queries=max_queries
        )
        
        # Search papers with limited results
        new_papers = {}
        for query in queries[:max_queries]:
            try:
                results = self.searcher.run(query)
                if results:
                    # Only take limited papers per query
                    for title, paper_data in list(results.items())[:max_papers_per_query]:
                        # Check if we already have this paper (by comparing bibtex keys)
                        bibtex = paper_data.get("bibtex", "")
                        match = re.search(r"@\w+\{(.+?),", bibtex)
                        if match:
                            bibtex_key = match.group(1).strip()
                            if bibtex_key not in existing_keys and title not in new_papers:
                                new_papers[title] = paper_data
                time.sleep(0.8)
            except Exception as e:
                print(f"[ERROR] Search failed for query '{query}': {e}")
        
        if not new_papers:
            print(f"[INFO] No new papers found for {section}")
            return None

        print(f"[Citation] Found {len(new_papers)} new papers for {section}")

        # Add new papers to global references
        for title, paper_data in new_papers.items():
            if title not in self.references:
                self.references[title] = paper_data

        # Format paper context
        paper_context = self._format_paper_context(new_papers)
        
        # Use prompt from YAML
        embed_prompt = self.prompts.add_new_citations_prompt.format(
            section=section,
            section_content=original_content,
            paper_context=paper_context,
            num_papers=len(new_papers),
        )

        enriched_content, _ = get_response_from_llm(
            msg=embed_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.citation_system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"Add_Citations_{section}",
        )

        # Verify new citations were added
        new_citations = set(re.findall(r'\\cite\{([^\}]+)\}', enriched_content))
        new_keys = set()
        for cite in new_citations:
            new_keys.update([k.strip() for k in cite.split(',')])
        
        added_count = len(new_keys - existing_keys)
        print(f"[Citation] Added {added_count} new citation(s) to {section}")
        
        return enriched_content
    
    def _enrich_section_with_citations(self, section: str, idea: Dict[str, Any]) -> Optional[str]:
        """
        Enrich section with citations:
        1. Generate queries → search papers (get abstract + bibtex)
        2. Use paper abstracts to guide LLM to add citations
        3. Replace paper titles with bibtex keys
        """
        original_content = self.generated_sections.get(section, "")
        if not original_content or len(original_content) < 50:
            return None

        # 1) Generate queries and search papers
        queries = self._generate_search_queries(
            idea, section=section, content_snippet=original_content[:600], max_queries=4
        )
        papers = self._search_papers_by_queries(queries)
        
        if not papers:
            print(f"[INFO] No papers found for {section}, skipping citation enrichment")
            return None

        # 2) Add papers to global references
        for title, paper_data in papers.items():
            if title not in self.references:
                self.references[title] = paper_data

        # 3) Format paper context with abstracts
        paper_context = self._format_paper_context(papers)
        
        # 4) Ask LLM to embed citations using paper titles
        embed_prompt = self.prompts.embed_citation_prompt.format(
            section=section,
            section_content=original_content,
            references=paper_context,
        )

        enriched_content, _ = get_response_from_llm(
            msg=embed_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.citation_system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"Embed_Citation_{section}",
        )

        # 5) Replace paper titles with bibtex keys
        final_content = self._replace_titles_with_bibtex_keys(enriched_content, papers)
        
        return final_content

    def _format_authors(self, authors: Any) -> str:
        """Format authors field which can be a string, list of dicts, or list of strings."""
        if isinstance(authors, str):
            return authors
        elif isinstance(authors, list):
            if not authors:
                return "Unknown authors"
            # Check if it's a list of dicts (Semantic Scholar format)
            if isinstance(authors[0], dict):
                author_names = [a.get("name", "") for a in authors if a.get("name")]
                if len(author_names) <= 3:
                    return " and ".join(author_names)
                else:
                    return f"{author_names[0]} et al."
            # List of strings
            elif isinstance(authors[0], str):
                if len(authors) <= 3:
                    return " and ".join(authors)
                else:
                    return f"{authors[0]} et al."
        return "Unknown authors"

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        """
        Generate Related Work section:
        1. Generate queries from idea → search papers (get abstract + bibtex)
        2. Use paper abstracts to write Related Work
        3. Replace paper titles with bibtex keys
        """
        # 1) Generate queries and search papers
        queries = self._generate_search_queries(idea, section="Related_Work", max_queries=6)
        papers = self._search_papers_by_queries(queries)
        
        if not papers:
            print("[WARNING] No papers found for Related Work")
            return

        # 2) Store papers in global references
        self.references = papers
        
        # 3) Format paper context with abstracts for LLM
        paper_context = self._format_paper_context(papers)
        
        # 4) Generate Related Work content using paper abstracts
        experiment = idea.get("Experiment", "No experiment details provided")
        related_work_prompt = self.prompts.related_work_prompt.format(
            related_work_tips=self.prompts.section_tips["Related_Work"],
            experiment=experiment,
            references=paper_context,
        )

        relatedwork_content, _ = get_response_from_llm(
            msg=related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt_related_work,
            cost_tracker=self.cost_tracker,
            task_name="Related_Work",
        )

        # Debug: Print citations to verify LLM used correct keys
        import re as debug_re
        citations = debug_re.findall(r'\\cite\{([^\}]+)\}', relatedwork_content)
        print(f"[DEBUG] Related Work citations found: {citations}")
        print(f"[DEBUG] Number of papers in references: {len(papers)}")
        
        # Extract expected bibtex keys from papers
        expected_keys = []
        for title, paper_data in papers.items():
            bibtex = paper_data.get("bibtex", "")
            bibtex_key_match = debug_re.search(r"@\w+\{(.+?),", bibtex)
            if bibtex_key_match:
                expected_keys.append(bibtex_key_match.group(1).strip())
        print(f"[DEBUG] Expected bibtex keys: {expected_keys[:6]}")
        
        # Check if all citations match expected keys
        citation_keys = [k.strip() for cite in citations for k in cite.split(',')]
        unmatched = [k for k in citation_keys if k not in expected_keys]
        if unmatched:
            print(f"[WARNING] Unmatched citation keys (may be removed later): {unmatched}")
        
        self.generated_sections["Related_Work"] = relatedwork_content

    def _refine_section(self, section: str) -> None:
        """Refine a section of the paper."""
        refinement_prompt = (
            self.prompts.refinement_prompt.format(
                section=section,
                section_tips=self.prompts.section_tips[section],
                section_content=self.generated_sections[section],
                error_list=self.prompts.error_list,
            )
            .replace(r"{{", "{")
            .replace(r"}}", "}")
        )

        refined_section, _ = get_response_from_llm(
            msg=refinement_prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"Refine {section}",
        )

        self.generated_sections[section] = refined_section

    def _refine_paper(self, num_rounds: int = 2, add_citations: bool = True) -> None:
        """Multi-round refinement with progressive citation enrichment."""
        print(f"\n{'='*60}")
        print(f"Starting {num_rounds}-round refinement with citation enrichment")
        print(f"{'='*60}\n")
        
        # First refine the title based on full draft
        full_draft = "\n\n".join(
            [
                f"\\section{{{section}}}\n\n{content}"
                for section, content in self.generated_sections.items()
            ]
        )

        refined_title, _ = get_response_from_llm(
            msg=self.prompts.title_refinement_prompt.format(full_draft=full_draft),
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name="Title Refinement",
        )
        self.generated_sections["Title"] = refined_title

        # Define refinement priority (most important sections first)
        refinement_sections = [
            "Method",           # Highest priority - needs most detail
            "Introduction",
            "Experimental_Setup",
            "Results",
            "Discussion",
            "Conclusion",
        ]
        
        # Multi-round refinement
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'─'*60}")
            print(f"REFINEMENT ROUND {round_num}/{num_rounds}")
            print(f"{'─'*60}\n")
            
            for section in refinement_sections:
                if section not in self.generated_sections:
                    continue
                    
                print(f"[Round {round_num}] Refining {section}...")
                
                # Build FULL paper context (all sections)
                full_paper_sections = []
                section_order = ["Abstract", "Related_Work", "Introduction", "Method", 
                                "Experimental_Setup", "Results", "Discussion", "Conclusion"]
                
                for sec_name in section_order:
                    if sec_name in self.generated_sections and sec_name != section:
                        content = self.generated_sections[sec_name]
                        # Truncate very long sections but keep substantial context
                        if len(content) > 2000:
                            content = content[:2000] + "\n[...truncated for brevity...]"
                        full_paper_sections.append(f"## {sec_name}\n{content}")
                
                other_sections_context = "\n\n".join(full_paper_sections)
                
                # Round-specific refinement goals
                if round_num == 1:
                    focus = "Add mathematical rigor, expand technical details, improve structure"
                elif round_num == 2:
                    focus = "Deepen analysis, add design rationale, enhance clarity"
                else:
                    focus = "Polish writing, ensure coherence, strengthen arguments"
                
                # Method-specific instruction
                method_specific_instruction = (
                    'For Method: Add more \\paragraph{{}} blocks, equations, and technical depth' 
                    if section == 'Method' 
                    else 'Enhance technical detail and clarity'
                )
                
                # Use prompt from YAML
                refinement_prompt = self.prompts.multi_round_refinement_prompt.format(
                    section=section,
                    round_num=round_num,
                    total_rounds=num_rounds,
                    focus=focus,
                    section_content=self.generated_sections[section],
                    section_tips=self.prompts.section_tips.get(section, ""),
                    other_sections_context=other_sections_context,  # Full context now
                    method_specific_instruction=method_specific_instruction,
                    error_list=self.prompts.error_list,
                )

                # Count citations before refinement
                import re as citation_re
                original_citations = set(citation_re.findall(r'\\cite\{([^\}]+)\}', self.generated_sections[section]))
                original_keys = set()
                for cite in original_citations:
                    original_keys.update([k.strip() for k in cite.split(',')])
                
                refined_content, _ = get_response_from_llm(
                    msg=refinement_prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.system_prompt,
                    cost_tracker=self.cost_tracker,
                    task_name=f"Refine_R{round_num}_{section}",
                )
                
                # Check if citations were lost during refinement
                refined_citations = set(citation_re.findall(r'\\cite\{([^\}]+)\}', refined_content))
                refined_keys = set()
                for cite in refined_citations:
                    refined_keys.update([k.strip() for k in cite.split(',')])
                
                # Report citation changes
                gained_keys = refined_keys - original_keys
                lost_keys = original_keys - refined_keys
                
                if lost_keys or gained_keys:
                    print(f"[Round {round_num}] Citation changes in {section}:")
                    print(f"  Before: {len(original_keys)} citations")
                    print(f"  After:  {len(refined_keys)} citations")
                    if gained_keys:
                        print(f"  ✅ Gained: {len(gained_keys)} ({list(gained_keys)[:3]}{'...' if len(gained_keys) > 3 else ''})")
                    if lost_keys:
                        print(f"  ❌ Lost:   {len(lost_keys)} ({list(lost_keys)[:3]}{'...' if len(lost_keys) > 3 else ''})")
                
                if lost_keys:
                    # Strategy: Keep original content if too many citations lost
                    loss_percentage = len(lost_keys) / len(original_keys) if original_keys else 0
                    if loss_percentage > 0.3:  # If >30% citations lost, revert
                        print(f"[Round {round_num}] ❌ REVERTING: Too many citations lost ({loss_percentage:.1%})")
                        refined_content = self.generated_sections[section]  # Revert
                    else:
                        print(f"[Round {round_num}] ⚠️ Accepting refined content (loss acceptable: {loss_percentage:.1%})")
                
                self.generated_sections[section] = refined_content

                # Add citations after each refinement (if enabled)
                if add_citations and round_num <= num_rounds:
                    try:
                        print(f"[Round {round_num}] Adding citations to {section}...")
                        enriched = self._enrich_section_with_citations_v2(
                            section, 
                            max_queries=3 if round_num == 1 else 2,  # Fewer queries in later rounds
                            max_papers_per_query=2  # Fewer papers to avoid overwhelming
                        )
                        if enriched:
                            self.generated_sections[section] = enriched
                            print(f"[Round {round_num}] ✅ Citations added to {section}")
                        else:
                            print(f"[Round {round_num}] ⚠️ No new citations for {section}")
                    except Exception as e:
                        print(f"[Round {round_num}] ❌ Citation enrichment failed for {section}: {e}")
                        traceback.print_exc()
                
                time.sleep(0.5)  # Rate limiting
        
        print(f"\n{'='*60}")
        print(f"Refinement completed!")
        print(f"{'='*60}\n")

