import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .llm import extract_json_between_markers, get_response_from_llm
from .texer import Texer


class Writer:
    def __init__(
        self,
        model: str,
        client: Any,
        base_dir: str,
        coder: Any,
        s2_api_key: Optional[str] = None
    ):
        """Initialize the PaperWriter with model and configuration."""
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.coder = coder
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")
        self.generated_sections: Dict[str, str] = {}

        # Load prompts
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "writer_prompt.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)
            
        # Initialize the Texer for LaTeX handling
        self.texer = Texer(model, client, base_dir, self.prompts)

    def perform_writeup(
        self,
        idea: Dict[str, Any],
        folder_name: str,
        template: str,
        num_cite_rounds: int = 20,
        engine: str = "semanticscholar"
    ) -> None:
        """Perform complete paper writeup process."""

        with open(os.path.join(folder_name, "experiment.py"), "r") as f:
            code = f.read()
        # extract experiment result from baseline_results.txt
        with open(os.path.join(folder_name, "baseline_results.txt"), "r") as f:
            baseline_result = f.read()
        # extract experiment result from experiment_results.txt
        with open(os.path.join(folder_name, "experiment_results.txt"), "r") as f:
            experiment_result = f.read()

        self.generated_sections = {}
        self._write_abstract(idea)

        for section in [
            "Introduction",
            "Background",
            "Method",
            "Experimental Setup",
            "Results",
            "Conclusion",
        ]:
            self._write_section(idea, code, baseline_result, experiment_result, section)

        self._write_related_work(idea)
        self._add_citations(num_cite_rounds, engine)
        self._refine_paper()

        name = idea.get("Title", "Research Paper")
        self.texer.generate_latex(
            output_pdf_path=f"{self.base_dir}/{name}.pdf", 
            template=template, 
            name=name,
            generated_sections=self.generated_sections
        )

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        """Write the abstract section."""

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
        """Write an individual section of the paper."""
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
        elif section in ["Background"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"][section]
            )
        elif section in ["Method", "Experimental Setup"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"]["Experimental Setup"],
                experiment = experiment,
                code = code
            )
        elif section in ["Results"]:
            section_prompt = self.prompts["section_prompt"][section].format(
                section_tips=self.prompts["section_tips"]["Results"],
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

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        """Write the related work section."""
        experiment = idea.get("Experiment", "No experiment details provided")

        related_work_prompt = self.prompts["related_work_prompt"].format(
            related_work_tips=self.prompts["section_tips"]["Related Work"],
            experiment = experiment
        )

        relatedwork_content, _ = get_response_from_llm(
            msg = related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts["write_system_prompt"]
        )

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
        """Perform second refinement of the entire paper."""
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
            "Related Work",
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
        print(self.generated_sections.keys())
        print('FINISHED REFINING SECTIONS')

    def _add_citations(self, num_cite_rounds: int, engine: str) -> None:
        """Add citations to the paper."""
        citations = self.generated_sections.get("Citations", "")

        for i in range(num_cite_rounds):
            # Use the current citations block as context for generating new citation suggestions.
            prompt, bibtex_string, done = self._get_citation_prompt(citations, i + 1, num_cite_rounds, engine)
            if done:
                print("No more citations needed.")
                break

            if prompt is not None:
                # Use the LLM to generate a citation response
                citation_response, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts["citation_system_prompt"].format(total_rounds=num_cite_rounds)
                )

                print("Bibtex string:", bibtex_string)

                if bibtex_string is not None:
                    citations += "\n" + bibtex_string
                print(f"Citations updated for round {i+1}.")
            else:
                # Handle TOO MANY REQUESTS ERROR
                print("No more citations needed.")
                break
        self.generated_sections["Citations"] = citations

    def _get_citation_prompt(
        self,
        draft: str,
        current_round: int,
        total_rounds: int,
        engine: str
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Get a prompt for citation suggestions."""
        msg_history: List[Dict[str, Any]] = []
        try:
            # Get initial citation suggestion
            text, msg_history = get_response_from_llm(
                self.prompts["citation_first_prompt"].format(
                    draft=draft,
                    current_round=current_round,
                    total_rounds=total_rounds
                ),
                client=self.client,
                model=self.model,
                system_message=self.prompts["citation_system_prompt"].format(total_rounds=total_rounds),
                msg_history=msg_history,
            )

            if "No more citations needed" in text:
                print("No more citations needed.")
                return None, None, True

            json_output = extract_json_between_markers(text)
            if not json_output:
                return None, None, False

            query = json_output["Query"]
            papers = self._search_for_papers(query, engine=engine)
            if not papers:
                print("No papers found.")
                return None, None, False

            # Get paper selection
            papers_str = self._format_paper_results(papers)
            text, msg_history = get_response_from_llm(
                self.prompts["citation_second_prompt"].format(
                    papers=papers_str,
                    current_round=current_round,
                    total_rounds=total_rounds,
                ),
                client=self.client,
                model=self.model,
                system_message=self.prompts["citation_system_prompt"].format(total_rounds=total_rounds),
                msg_history=msg_history,
            )

            if "Do not add any" in text:
                print("Do not add any.")
                return None, None, False

            json_output = extract_json_between_markers(text)
            if not json_output:
                return None, None, False

            desc = json_output["Description"]
            selected_papers = json_output["Selected"]

            if selected_papers == "[]":
                return None, None, False

            selected_indices = list(map(int, selected_papers.strip("[]").split(",")))
            if not all(0 <= i < len(papers) for i in selected_indices):
                return None, None, False

            # Get bibtex entries for selected papers
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]
            bibtex_string = "\n".join(bibtexs)

            final_citation = self.prompts["citation_aider_format"].format(
                bibtex=bibtex_string,
                description=desc
            )
            return final_citation, bibtex_string, False
        except Exception as e:
            print(f"Error in getting citation prompt: {e}")
            return None, None, False