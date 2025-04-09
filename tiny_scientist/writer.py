import json
import os
import os.path as osp
import re
import time
import traceback
from typing import Any, Dict, List, Optional

import yaml

from .format import ACLFormat, BaseFormat, ICLRFormat
from .llm import extract_json_between_markers, get_response_from_llm
from .tool import BaseTool, PaperSearchTool


class Writer:
    def __init__(self, model: str, client: Any, base_dir: str, template: str,
                 temperature: float = 0.75, s2_api_key: Optional[str] = None):
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.template = template
        self.temperature = temperature
        self.searcher: BaseTool = PaperSearchTool()
        self.formatter: BaseFormat
        if self.template == 'acl':
            self.formatter = ACLFormat(self.client, self.model)
        elif self.template == 'iclr':
            self.formatter = ICLRFormat(self.client, self.model)

        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "writer_prompt.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def run(self, idea: Dict[str, Any], folder_name: str) -> None:
        with open(osp.join(folder_name, "experiment.py"), "r") as f:
            code = f.read()

        with open(osp.join(folder_name, "baseline_results.txt"), "r") as f:
            baseline_result = f.read()

        with open(osp.join(folder_name, "experiment_results.txt"), "r") as f:
            experiment_result = f.read()

        self.generated_sections: Dict[str, Any] = {}
        self.references: Dict[str, Any] = {}

        self._write_abstract(idea)

        for section in [
            "Introduction",
            "Method",
            "Experimental Setup",
            "Results",
            "Discussion",
            "Conclusion",
        ]:
            self._write_section(idea, code, baseline_result, experiment_result, section)

        self._write_related_work(idea)
        self._refine_paper()
        self._add_citations(idea)

        paper_name = idea.get("Title", "Research Paper")

        self.formatter.run(self.generated_sections, self.references,
                           self.base_dir, f"{self.base_dir}/{paper_name}.pdf", paper_name)

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        title = idea.get("Title", "Research Paper")
        experiment = idea.get("Experiment", "No experiment details provided")

        abstract_prompt = self.prompts["abstract_prompt"].format(
            abstract_tips=self.prompts["section_tips"]["Abstract"],
            title = title,
            experiment = experiment
        )

        abstract_content, _ = get_response_from_llm(
            msg = abstract_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts["write_system_prompt"]
        )

        self.generated_sections["Abstract"] = abstract_content


    def _write_section(self,
                       idea: Dict[str, Any],
                       code: str,
                       baseline_result: str,
                       experiment_result: str,
                       section: str) -> None:

        title = idea.get("Title", "Research Paper")
        experiment = idea.get("Experiment", "No experiment details provided")

        if section in ["Introduction"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"][section],
                title = title,
                experiment = experiment
            )
        elif section in ["Conclusion"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"][section],
                experiment = experiment
            )
        elif section in ["Method", "Experimental Setup"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"][section],
                experiment = experiment,
                code = code
            )
        elif section in ["Results", "Discussion"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"][section],
                experiment = experiment,
                baseline_results = baseline_result,
                experiment_results = experiment_result
            )


        section_content, _ = get_response_from_llm(
            msg = section_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts["write_system_prompt"]
        )

        self.generated_sections[section] = section_content

    def _get_citations_related_work(self,
                                    idea: Dict[str, Any],
                                    num_cite_rounds: int,
                                    total_num_papers: int
                                    ) -> List[str]:

        idea_title = idea.get("Title", "Research Paper")
        experiment = idea.get("Experiment", "No experiment details provided")

        num_papers = total_num_papers // num_cite_rounds if num_cite_rounds > 0 else total_num_papers
        collected_papers: List[str] = []

        for round_num in range(num_cite_rounds):
            prompt = self.prompts["citation_related_work_prompt"].format(
                idea_title=idea_title,
                experiment=experiment,
                num_papers = num_papers,
                round_num=round_num+1,
                collected_papers = collected_papers,
                total_rounds=num_cite_rounds
            )
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts["citation_system_prompt"]
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
        citations = self._get_citations_related_work(idea, num_cite_rounds=2, total_num_papers=10)

        paper_source =  self._search_reference(citations)
        self.references = paper_source

        reference_list = "\n".join([f"- {title}" for title in paper_source.keys()])
        reference_list = reference_list.replace("{", "{{").replace("}", "}}")

        experiment = idea.get("Experiment", "No experiment details provided")

        related_work_prompt = self.prompts["related_work_prompt"].format(
            related_work_tips=self.prompts["section_tips"]["Related Work"],
            experiment = experiment,
            references = reference_list,
        )

        relatedwork_content, _ = get_response_from_llm(
            msg = related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts["write_system_prompt_related_work"]
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
                        relatedwork_content
                    )
                except Exception:
                    print(f"[ERROR] Failed to replace citation for title: {title}")
                    traceback.print_exc()

        self.generated_sections["Related Work"] = relatedwork_content

    def _refine_section(self, section: str) -> None:
        """Refine a section of the paper."""
        refinement_prompt = self.prompts["refinement_prompt"].format(
            section = section,
            section_content=self.generated_sections[section],
            error_list=self.prompts["error_list"]
        ).replace(r"{{", "{").replace(r"}}", "}")

        refined_section, _ = get_response_from_llm(
            msg = refinement_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts["write_system_prompt"]
        )

        self.generated_sections[section] = refined_section

    def _refine_paper(self) -> None:
        full_draft = "\n\n".join(
            [f"\\section{{{section}}}\n\n{content}" for section, content in self.generated_sections.items()]
        )

        refined_title, _ = get_response_from_llm(
            msg = self.prompts["title_refinement_prompt"].format(
                full_draft = full_draft
            ),
            client=self.client,
            model=self.model,
            system_message=self.prompts["write_system_prompt"]
        )

        self.generated_sections["Title"] = refined_title

        for section in [
            "Abstract",
            "Introduction",
            "Background",
            "Method",
            "Experimental Setup",
            "Results",
            "Conclusion"
        ]:
            if section in self.generated_sections.keys():
                print(f"REFINING SECTION: {section}")
                second_refinement_prompt = self.prompts["second_refinement_prompt"].format(
                    section = section,
                    tips=self.prompts["section_tips"][section],
                    full_draft = full_draft,
                    section_content=self.generated_sections[section],
                    error_list=self.prompts["error_list"]
                ).replace(r"{{", "{").replace(r"}}", "}")

                refined_section, _ = get_response_from_llm(
                    msg = second_refinement_prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts["write_system_prompt"]
                )

                self.generated_sections[section] = refined_section

    def _add_citations(self, idea: Dict[str, Any]) -> None:

        idea_title = idea.get("Title", "Research Paper")
        experiment = idea.get("Experiment", "No experiment details provided")

        for section in [
            "Introduction",
            "Method",
            "Experimental Setup",
            "Discussion"
        ]:
            if section in self.generated_sections.keys():
                try:
                    original_content = self.generated_sections[section]
                    collected_papers = []

                    add_citation_prompt = self.prompts["add_citation_prompt"].format(
                        idea_title=idea_title,
                        experiment=experiment,
                        section = section,
                        section_content=original_content,
                    )

                    response, _ = get_response_from_llm(
                        msg = add_citation_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts["citation_system_prompt"]
                    )

                    try:
                        new_titles = json.loads(response)
                    except json.JSONDecodeError:
                        new_titles = extract_json_between_markers(response)

                    collected_papers.extend(new_titles)
                    paper_source =  self._search_reference(collected_papers)

                    if not paper_source:
                        continue

                    for title, entry in paper_source.items():
                        if title not in self.references:
                            self.references[title] = entry

                    reference_list = "\n".join([f"- {title}" for title in paper_source.keys()])
                    reference_list = reference_list.replace("{", "{{").replace("}", "}}")

                    embed_citation_prompt = self.prompts["embed_citation_prompt"].format(
                        section=section,
                        section_content=original_content,
                        references=reference_list
                    )

                    refined_section, _ = get_response_from_llm(
                        msg=embed_citation_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts["citation_system_prompt"]
                    )

                    print(f"Refined section for {section}: {refined_section}")

                    for title, meta in paper_source.items():

                        match = re.search(r"@\w+\{(.+?),", meta.get("bibtex", ""))
                        if match:
                            bibtex_key = match.group(1)
                            escaped_title = re.escape(title)
                            pattern = r"\\cite\{\s*" + escaped_title + r"\s*\}"
                            refined_section = re.sub(
                                                    pattern,
                                                    lambda _: f"\\cite{{{bibtex_key}}}",
                                                    refined_section
                                                )
                    self.generated_sections[section] = refined_section

                except Exception:
                    print(f"[ERROR] Failed to add citations to section: {section}")
                    traceback.print_exc()
