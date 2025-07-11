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
    ) -> None:
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.template = template
        self.temperature = temperature
        self.searcher: BaseTool = PaperSearchTool(s2_api_key=s2_api_key)
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

        self._write_abstract(idea)

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

        for section in sections:
            self._write_section(idea, code, experiment_result, section, baseline_result)

        self._write_related_work(idea)
        self._refine_paper()

        self._add_citations(idea)
        self._generate_diagram_for_section()

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
        title = idea.get("Title", "Research Paper")

        abstract_prompt = self.prompts.abstract_prompt.format(
            abstract_tips=self.prompts.section_tips["Abstract"],
            title=title,
            problem=idea["Problem"],
            importance=idea["Importance"],
            difficulty=idea["Difficulty"],
            novelty=idea["NoveltyComparison"],
            experiment=idea["Experiment"],
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

    def _write_section(
        self,
        idea: Dict[str, Any],
        code: str,
        experiment_result: str,
        section: str,
        baseline_result: Optional[str] = "",
    ) -> None:
        title = idea.get("Title", "Research Paper")
        experiment = idea.get("Experiment")
        print(f"Writing section: {section}...")

        if section in ["Introduction"]:
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                title=title,
                problem=idea["Problem"],
                importance=idea["Importance"],
                difficulty=idea["Difficulty"],
                novelty=idea["NoveltyComparison"],
                experiment=experiment,
            )
        elif section in ["Conclusion"]:
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                experiment=experiment,
            )
        elif section in ["Method", "Experimental_Setup"]:
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                problem=idea["Problem"],
                importance=idea["Importance"],
                difficulty=idea["Difficulty"],
                novelty=idea["NoveltyComparison"],
                experiment=experiment,
                code=code,
            )
        elif section in ["Results", "Discussion"]:
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                experiment=experiment,
                baseline_results=baseline_result,
                experiment_results=experiment_result,
            )
        elif section == "Analysis":
            # For non-experimental papers, use the research plan content
            research_plan = idea.get("ResearchPlan", experiment)
            section_prompt = self.prompts.section_prompt.get(
                section, self.prompts.section_prompt.get("Results", "")
            ).format(
                section_tips=self.prompts.section_tips.get(
                    section, self.prompts.section_tips.get("Results", "")
                ),
                experiment=research_plan,
                baseline_results=baseline_result,
                experiment_results=experiment_result,
            )

        section_content, _ = get_response_from_llm(
            msg=section_prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_prompt,
            cost_tracker=self.cost_tracker,
            task_name=f"{section} section",
        )

        self.generated_sections[section] = section_content

    def _get_citations_related_work(
        self, idea: Dict[str, Any], num_cite_rounds: int, total_num_papers: int
    ) -> List[str]:
        idea_title = idea.get("Title", "Research Paper")

        num_papers = (
            total_num_papers // num_cite_rounds
            if num_cite_rounds > 0
            else total_num_papers
        )
        collected_papers: List[str] = []

        for round_num in range(num_cite_rounds):
            prompt = self.prompts.citation_related_work_prompt.format(
                idea_title=idea_title,
                problem=idea["Problem"],
                num_papers=num_papers,
                round_num=round_num + 1,
                collected_papers=collected_papers,
                total_rounds=num_cite_rounds,
            )
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.citation_system_prompt,
                cost_tracker=self.cost_tracker,
                task_name="Related Work",
            )

            try:
                new_titles = json.loads(response)
            except json.JSONDecodeError:
                new_titles = extract_json_between_markers(response)

            if new_titles and isinstance(new_titles, list):
                collected_papers.extend(new_titles)
            else:
                print(f"Round {round_num+1}: No valid titles returned.")

            if len(collected_papers) >= total_num_papers:
                break
            time.sleep(1)

        return collected_papers

    def _search_reference(self, paper_list: List[str]) -> Dict[str, Any]:
        results_dict = {}

        for paper_name in paper_list:
            try:
                result = self.searcher.run(paper_name)

                if result:
                    if paper_name in result:
                        results_dict[paper_name] = result[paper_name]
                    else:
                        first_key = next(iter(result))
                        results_dict[first_key] = result[first_key]

                time.sleep(1.0)
            except Exception as e:
                print(f"[ERROR] While processing '{paper_name}': {e}")
                traceback.print_exc()

        return results_dict

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        citations = self._get_citations_related_work(
            idea, num_cite_rounds=2, total_num_papers=10
        )

        paper_source = self._search_reference(citations)
        self.references = paper_source

        reference_list = "\n".join([f"- {title}" for title in paper_source.keys()])
        reference_list = reference_list.replace("{", "{{").replace("}", "}}")

        experiment = idea.get("Experiment", "No experiment details provided")

        related_work_prompt = self.prompts.related_work_prompt.format(
            related_work_tips=self.prompts.section_tips["Related_Work"],
            experiment=experiment,
            references=reference_list,
        )

        relatedwork_content, _ = get_response_from_llm(
            msg=related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt_related_work,
            cost_tracker=self.cost_tracker,
            task_name="Related Work",
        )

        for title, meta in paper_source.items():
            match = re.search(r"@\w+\{(.+?),", meta.get("bibtex", ""))
            if match:
                try:
                    bibtex_key = match.group(1)
                    escaped_title = re.escape(title)
                    pattern = r"\\cite\{\s*" + escaped_title + r"\s*\}"
                    relatedwork_content = re.sub(
                        pattern,
                        lambda _: f"\\cite{{{bibtex_key}}}",
                        relatedwork_content,
                    )
                except Exception:
                    print(f"[ERROR] Failed to replace citation for title: {title}")
                    traceback.print_exc()

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

    def _refine_paper(self) -> None:
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

        for section in [
            "Introduction",
            "Background",
            "Method",
            "Experimental_Setup",
            "Results",
            "Conclusion",
        ]:
            if section in self.generated_sections.keys():
                print(f"Refining section: {section}...")
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

                refined_section_content, _ = get_response_from_llm(
                    msg=refinement_prompt,
                    client=self.client,
                    model=self.model,
                    system_message="",
                    cost_tracker=self.cost_tracker,
                    task_name=f"Second Refine {section}",
                )

                self.generated_sections[section] = refined_section_content

    def _add_citations(self, idea: Dict[str, Any]) -> None:
        idea_title = idea.get("Title", "Research Paper")

        for section in ["Introduction", "Method", "Experimental_Setup", "Discussion"]:
            if section in self.generated_sections.keys():
                try:
                    original_content = self.generated_sections[section]
                    collected_papers = []

                    add_citation_prompt = self.prompts.add_citation_prompt.format(
                        idea_title=idea_title,
                        problem=idea["Problem"],
                        importance=idea["Importance"],
                        challenges=idea["Difficulty"],
                        section=section,
                        section_content=original_content,
                    )

                    response, _ = get_response_from_llm(
                        msg=add_citation_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts.citation_system_prompt,
                        cost_tracker=self.cost_tracker,
                        task_name=f"Add Citation to {section}",
                    )

                    try:
                        new_titles = json.loads(response)
                    except json.JSONDecodeError:
                        new_titles = extract_json_between_markers(response)

                    collected_papers.extend(new_titles)
                    paper_source = self._search_reference(collected_papers)

                    if not paper_source:
                        continue

                    for title, entry in paper_source.items():
                        if title not in self.references:
                            self.references[title] = entry

                    reference_list = "\n".join(
                        [f"- {title}" for title in paper_source.keys()]
                    )
                    reference_list = reference_list.replace("{", "{{").replace(
                        "}", "}}"
                    )

                    embed_citation_prompt = self.prompts.embed_citation_prompt.format(
                        section=section,
                        section_content=original_content,
                        references=reference_list,
                    )

                    refined_section, _ = get_response_from_llm(
                        msg=embed_citation_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts.citation_system_prompt,
                        cost_tracker=self.cost_tracker,
                        task_name=f"Embed Citation in {section}",
                    )

                    for title, meta in paper_source.items():
                        match = re.search(r"@\w+\{(.+?),", meta.get("bibtex", ""))
                        if match:
                            bibtex_key = match.group(1)
                            escaped_title = re.escape(title)
                            pattern = r"\\cite\{\s*" + escaped_title + r"\s*\}"
                            refined_section = re.sub(
                                pattern,
                                lambda _: f"\\cite{{{bibtex_key}}}",
                                refined_section,
                            )
                    self.generated_sections[section] = refined_section

                except Exception:
                    print(f"[ERROR] Failed to add citations to section: {section}")
                    traceback.print_exc()
