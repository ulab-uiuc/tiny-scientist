import os
import os.path as osp
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import json
from typing import Any, Dict, List, Optional, Tuple

import backoff
import pyalex
import requests
import yaml
from pyalex import Works
from PyPDF2 import PageObject, PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .tool import PaperSearchTool
from .format import ACLFormat, ICLRFormat

from .llm import extract_json_between_markers, get_response_from_llm


class Writer:
    def __init__(self, model: str, client: any, base_dir: str, template: str, config_dir: str,
                 temperature: float = 0.75, s2_api_key: Optional[str] = None):
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.template = template
        self.temperature = temperature
        self.searcher = PaperSearchTool()
        if self.template == 'acl':
            self.formatter = ACLFormat(self.client, self.model)
        elif self.template == 'iclr':
            self.formatter = ICLRFormat(self.client, self.model)

        with open(osp.join(config_dir, "writer_prompt.yaml"), "r") as f:
            self.prompts = yaml.safe_load(f)

    def perform_writeup(self, idea: Dict[str, Any], folder_name: str, num_cite_rounds: int = 20) -> None:
        with open(osp.join(folder_name, "experiment.py"), "r") as f:
            code = f.read()

        with open(osp.join(folder_name, "baseline_results.txt"), "r") as f:
            baseline_result = f.read()

        with open(osp.join(folder_name, "experiment_results.txt"), "r") as f:
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
        self._refine_paper()

        paper_name = idea.get("Title", "Research Paper")

        self.formatter.run(self.generated_sections, 
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


    def _get_citations_related_work(self, 
                                    idea: Dict[str, Any], 
                                    num_cite_rounds: int, 
                                    total_num_papers: int
                                    ) -> List[str]:
        
        idea_title = idea.get("Title", "Research Paper")  
        experiment = idea.get("Experiment", "No experiment details provided")

        num_papers = total_num_papers // num_cite_rounds if num_cite_rounds > 0 else total_num_papers
        collected_papers = []

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

            print(f"Round {round_num+1} raw response:", response)

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
    
    def _search_reference(self, paper_list: List[str]) -> Optional[str]:
        for paper_name in paper_list:
            results = self.searcher.run(paper_name)
            print(results)
        

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        citations = self._get_citations_related_work(idea, num_cite_rounds=5, total_num_papers=20)
        # paper_source =  self._search_reference(citations)
        
        experiment = idea.get("Experiment", "No experiment details provided")

        citations_list = "\n".join([f"- {c}" for c in citations])
        escaped_citations_list = citations_list.replace("{", "{{").replace("}", "}}")

        related_work_prompt = self.prompts["related_work_prompt"].format(
            related_work_tips=self.prompts["section_tips"]["Related Work"],
            experiment = experiment,
            citations = escaped_citations_list,
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













# class Writer:
#     def __init__(
#         self,
#         model: str,
#         client: Any,
#         base_dir: str,
#         coder: Any,
#         s2_api_key: Optional[str] = None
#     ):
#         """Initialize the PaperWriter with model and configuration."""
#         self.model = model
#         self.client = client
#         self.base_dir = base_dir
#         self.coder = coder
#         self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")
#         self.generated_sections: Dict[str, str] = {}
#         self.searcher = PaperSearchTool()

#         # Load prompts
#         yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "writer_prompt.yaml")
#         with open(yaml_path, "r") as f:
#             self.prompts = yaml.safe_load(f)

#     def perform_writeup(
#         self,
#         idea: Dict[str, Any],
#         folder_name: str,
#         template: str,
#         num_cite_rounds: int = 20,
#         engine = 'semanticscholar'
#     ) -> None:
#         """Perform complete paper writeup process."""

#         with open(os.path.join(folder_name, "experiment.py"), "r") as f:
#             code = f.read()
#         # extract experiment result from baseline_results.txt
#         with open(os.path.join(folder_name, "baseline_results.txt"), "r") as f:
#             baseline_result = f.read()
#         # extract experiment result from experiment_results.txt
#         with open(os.path.join(folder_name, "experiment_results.txt"), "r") as f:
#             experiment_result = f.read()

#         self.generated_sections = {}
#         self._write_abstract(idea)

#         for section in [
#             "Introduction",
#             "Background",
#             "Method",
#             "Experimental Setup",
#             "Results",
#             "Conclusion",
#         ]:
#             self._write_section(idea, code, baseline_result, experiment_result, section)

#         self._write_related_work(idea)
#         self._refine_paper()

#         name = idea.get("Title", "Research Paper")
#         self.generate_latex(f"{self.base_dir}/{name}.pdf", template, name)

#     def _write_abstract(self, idea: Dict[str, Any]) -> None:
        
#         title = idea.get("Title", "Research Paper")
#         experiment = idea.get("Experiment", "No experiment details provided")

#         abstract_prompt = self.prompts["abstract_prompt"].format(
#             abstract_tips=self.prompts["section_tips"]["Abstract"],
#             title = title,
#             experiment = experiment
#         )

#         abstract_content, _ = get_response_from_llm(
#             msg = abstract_prompt,
#             client=self.client,
#             model=self.model,
#             system_message=self.prompts["write_system_prompt"]
#         )

#         # self._refine_section("Abstract")
#         self.generated_sections["Abstract"] = abstract_content

#     def _write_section(self,
#                        idea: Dict[str, Any],
#                        code: str,
#                        baseline_result: str,
#                        experiment_result: str,
#                        section: str) -> None:

#         title = idea.get("Title", "Research Paper")
#         experiment = idea.get("Experiment", "No experiment details provided")

#         if section in ["Introduction"]:
#             section_prompt = self.prompts["section_prompt"][section].format(
#                 section_tips=self.prompts["section_tips"][section],
#                 title = title,
#                 experiment = experiment
#             )
#         elif section in ["Conclusion"]:
#             section_prompt = self.prompts["section_prompt"][section].format(
#                 section_tips=self.prompts["section_tips"][section],
#                 experiment = experiment
#             )
#         elif section in ["Background"]:
#             section_prompt = self.prompts["section_prompt"][section].format(
#                 section_tips=self.prompts["section_tips"][section]
#             )
#         elif section in ["Method", "Experimental Setup"]:
#             section_prompt = self.prompts["section_prompt"][section].format(
#                 section_tips=self.prompts["section_tips"]["Experimental Setup"],
#                 experiment = experiment,
#                 code = code
#             )
#         elif section in ["Results"]:
#             section_prompt = self.prompts["section_prompt"][section].format(
#                 section_tips=self.prompts["section_tips"]["Results"],
#                 experiment = experiment,
#                 baseline_results = baseline_result,
#                 experiment_results = experiment_result
#             )

#         section_content, _ = get_response_from_llm(
#             msg = section_prompt,
#             client=self.client,
#             model=self.model,
#             system_message=self.prompts["write_system_prompt"]
#         )

#         self.generated_sections[section] = section_content

#     def _get_citations_related_work(self, 
#                                     idea: Dict[str, Any], 
#                                     num_cite_rounds: int, 
#                                     total_num_papers: int
#                                     ) -> List[str]:
        
#         idea_title = idea.get("Title", "Research Paper")  
#         experiment = idea.get("Experiment", "No experiment details provided")

#         num_papers = total_num_papers // num_cite_rounds if num_cite_rounds > 0 else total_num_papers
#         collected_papers = []

#         for round_num in range(num_cite_rounds):
#             prompt = self.prompts["citation_related_work_prompt"].format(
#                 idea_title=idea_title,
#                 experiment=experiment,
#                 num_papers = num_papers,
#                 round_num=round_num+1,
#                 collected_papers = collected_papers,
#                 total_rounds=num_cite_rounds
#             )
#             response, _ = get_response_from_llm(
#                 msg=prompt,
#                 client=self.client,
#                 model=self.model,
#                 system_message=self.prompts["citation_system_prompt"]
#             )

#             print(f"Round {round_num+1} raw response:", response)

#             try:
#                 new_titles = json.loads(response)
#             except json.JSONDecodeError:
#                 new_titles = extract_json_between_markers(response)

#             if new_titles and isinstance(new_titles, list):
#                 collected_papers.extend(new_titles)
#             else:
#                 print(f"Round {round_num+1}: No valid titles returned.")
  
#             if len(collected_papers) >= total_num_papers:
#                 break
#             time.sleep(1)

#         return collected_papers
    
#     def _search_reference(self, paper_list: List[str]) -> Optional[str]:
#         for paper_name in paper_list:
#             results = self.searcher.run(paper_name)
#             print(results)
        
#         exit()

        


#     def _write_related_work(self, idea: Dict[str, Any]) -> None:
#         citations = self._get_citations_related_work(idea, num_cite_rounds=5, total_num_papers=20)
#         paper_source =  self._search_reference(citations)
        
#         experiment = idea.get("Experiment", "No experiment details provided")

#         citations_list = "\n".join([f"- {c}" for c in citations])
#         escaped_citations_list = citations_list.replace("{", "{{").replace("}", "}}")

#         related_work_prompt = self.prompts["related_work_prompt"].format(
#             related_work_tips=self.prompts["section_tips"]["Related Work"],
#             experiment = experiment,
#             citations = escaped_citations_list,
#         )

#         relatedwork_content, _ = get_response_from_llm(
#             msg = related_work_prompt,
#             client=self.client,
#             model=self.model,
#             system_message=self.prompts["write_system_prompt"]
#         )

#         self.generated_sections["Related Work"] = relatedwork_content

#     def _refine_section(self, section: str) -> None:
#         """Refine a section of the paper."""
#         refinement_prompt = self.prompts["refinement_prompt"].format(
#             section = section,
#             section_content=self.generated_sections[section],
#             error_list=self.prompts["error_list"]
#         ).replace(r"{{", "{").replace(r"}}", "}")

#         refined_section, _ = get_response_from_llm(
#             msg = refinement_prompt,
#             client=self.client,
#             model=self.model,
#             system_message=self.prompts["write_system_prompt"]
#         )

#         self.generated_sections[section] = refined_section

#     def update_custom_bib(self, dest_template_dir: str, template: str) -> None:
#         all_keys = set()
#         citation_patterns = [
#             r"\\cite\{([^}]+)\}",
#             r"\\citep\{([^}]+)\}",
#             r"\\citet\{([^}]+)\}"
#         ]

#         for content in self.generated_sections.values():
#             for pattern in citation_patterns:
#                 matches = re.findall(pattern, content)
#                 for m in matches:
#                     keys = [key.strip() for key in m.split(",")]
#                     all_keys.update(keys)

#         if template == 'acl':
#             bib_path = osp.join(dest_template_dir, "latex", "custom.bib")
#         if template == 'iclr':
#             # you should create a custom.bib file in the iclr folder
#             bib_path = osp.join(dest_template_dir, "custom.bib")

#         if osp.exists(bib_path):
#             with open(bib_path, "r", encoding="utf-8") as f:
#                 bib_content = f.read()
#             existing_keys = set(re.findall(r"@.+?\{([^,]+),", bib_content))
#         else:
#             bib_content = ""
#             existing_keys = set()

#         # 3. Find missing keys
#         missing_keys = all_keys - existing_keys
#         if not missing_keys:
#             print("All citation keys are already present in custom.bib.")
#             return

#         # 4. For each missing key, get the bibtex entry
#         new_entries = []
#         for key in missing_keys:
#             bibtex_entry = self._get_bibtex_for_key(key)
#             if bibtex_entry:
#                 new_entries.append(bibtex_entry)
#             else:
#                 print(f"Warning: Could not retrieve bibtex for key '{key}'.")

#         # 5. Append the new entries to custom.bib
#         if new_entries:
#             updated_bib = bib_content + "\n" + "\n".join(new_entries)
#             with open(bib_path, "w", encoding="utf-8") as f:
#                 f.write(updated_bib)
#             print(f"Updated custom.bib with entries for: {', '.join(missing_keys)}")
#         else:
#             print("No new bibtex entries were added.")

#     def _get_bibtex_for_key(self, key: str) -> Optional[str]:
#         prompt = f"Provide the bibtex entry for the paper with citation key '{key}'. Output only the bibtex entry."
#         bibtex_entry, _ = get_response_from_llm(
#             msg=prompt,
#             client=self.client,
#             model=self.model,
#             system_message="You are an expert in academic citations. Please provide a valid bibtex entry."
#         )
#         # A simple check: ensure it contains an @ and the key appears in the entry.
#         if "@" in bibtex_entry and key in bibtex_entry:
#             return bibtex_entry.strip()
#         else:
#             return None

#     def _refine_paper(self) -> None:
#         full_draft = "\n\n".join(
#             [f"\\section{{{section}}}\n\n{content}" for section, content in self.generated_sections.items()]
#         )

#         refined_title, _ = get_response_from_llm(
#             msg = self.prompts["title_refinement_prompt"].format(
#                 full_draft = full_draft
#             ),
#             client=self.client,
#             model=self.model,
#             system_message=self.prompts["write_system_prompt"]
#         )

#         self.generated_sections["Title"] = refined_title

#         for section in [
#             "Abstract",
#             "Related Work",
#             "Introduction",
#             "Background",
#             "Method",
#             "Experimental Setup",
#             "Results",
#             "Conclusion"
#         ]:
#             if section in self.generated_sections.keys():
#                 print(f"REFINING SECTION: {section}")
#                 second_refinement_prompt = self.prompts["second_refinement_prompt"].format(
#                     section = section,
#                     tips=self.prompts["section_tips"][section],
#                     full_draft = full_draft,
#                     section_content=self.generated_sections[section],
#                     error_list=self.prompts["error_list"]
#                 ).replace(r"{{", "{").replace(r"}}", "}")

#                 refined_section, _ = get_response_from_llm(
#                     msg = second_refinement_prompt,
#                     client=self.client,
#                     model=self.model,
#                     system_message=self.prompts["write_system_prompt"]
#                 )

#                 self.generated_sections[section] = refined_section
#         print(self.generated_sections.keys())
#         print('FINISHED REFINING SECTIONS')

#     @backoff.on_exception(
#         backoff.expo,
#         requests.exceptions.HTTPError,
#         on_backoff=lambda details: print(
#             f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
#             f"calling function {details['target'].__name__} at {time.strftime('%X')}"
#         )
#     )

#     def _search_for_papers(
#         self,
#         query: str,
#         result_limit: int = 10,
#         engine: str = "semanticscholar"
#     ) -> Optional[List[Dict[str, Any]]]:
#         """Search for papers using specified search engine."""
#         if not query:
#             return None

#         if engine == "semanticscholar":
#             return self._search_semanticscholar(query, result_limit)
#         elif engine == "openalex":
#             return self._search_openalex(query, result_limit)
#         else:
#             raise NotImplementedError(f"{engine=} not supported!")

#     def _search_semanticscholar(
#         self,
#         query: str,
#         result_limit: int
#     ) -> Optional[List[Dict[str, Any]]]:
#         """Search papers using Semantic Scholar API."""
#         rsp = requests.get(
#             "https://api.semanticscholar.org/graph/v1/paper/search",
#             headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
#             params={
#                 "query": query,
#                 "limit": result_limit,
#                 "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
#             },
#         )
#         print(f"Response Status Code: {rsp.status_code}")
#         print(f"Response Content: {rsp.text[:500]}")
#         rsp.raise_for_status()

#         results = rsp.json()
#         total = results["total"]
#         time.sleep(1.0)

#         return results["data"] if total else None

#     def _search_openalex(
#         self,
#         query: str,
#         result_limit: int
#     ) -> Optional[List[Dict[str, Any]]]:
#         """Search papers using OpenAlex API."""

#         mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
#         if mail:
#             pyalex.config.email = mail
#         else:
#             print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")

#         works = Works().search(query).get(per_page=result_limit)
#         if not works:
#             return None

#         papers = []
#         for work in works:
#             venue = "Unknown"
#             for location in work["locations"]:
#                 if location["source"] is not None:
#                     potential_venue = location["source"]["display_name"]
#                     if potential_venue:
#                         venue = potential_venue
#                         break

#             authors_list = [
#                 author["author"]["display_name"]
#                 for author in work["authorships"]
#             ]
#             authors = (
#                 " and ".join(authors_list)
#                 if len(authors_list) < 20
#                 else f"{authors_list[0]} et al."
#             )

#             abstract = work["abstract"] or ""
#             if len(abstract) > 1000:
#                 print(
#                     f"[WARNING] {work['title']}: Abstract length {len(abstract)} is too long! "
#                     f"Using first 1000 chars."
#                 )
#                 abstract = abstract[:1000]

#             papers.append({
#                 "title": work["title"],
#                 "authors": authors,
#                 "venue": venue,
#                 "year": work["publication_year"],
#                 "abstract": abstract,
#                 "citationCount": work["cited_by_count"],
#             })

#         return papers

#     @staticmethod
#     def _format_paper_results(papers: Optional[List[Dict[str, Any]]]) -> str:
#         """Format paper results into a string."""
#         if not papers:
#             return "No papers found."

#         paper_strings = []
#         for i, paper in enumerate(papers):
#             paper_strings.append(
#                 f"{i}: {paper['title']}. {paper['authors']}. {paper['venue']}, {paper['year']}.\n"
#                 f"Abstract: {paper['abstract']}"
#             )
#         return "\n\n".join(paper_strings)

#     def clean_latex_content(self, content: str) -> str:
#         match = re.search(r'```latex\s*(.*?)\s*```', content, flags=re.DOTALL)
#         if match:
#             return match.group(1)

#         # If no code block is found, perform minimal cleaning:
#         lines = content.splitlines()
#         cleaned_lines = []
#         for line in lines:
#             stripped = line.strip()
#             # Remove lines that are exactly code fences (```), but keep inline backticks if any.
#             if stripped in ["```"]:
#                 continue
#             # Remove markdown header lines (starting with '#' and not a LaTeX comment)
#             if stripped.startswith("#") and not stripped.startswith("%"):
#                 continue
#             cleaned_lines.append(line)
#         return "\n".join(cleaned_lines)

#     def insert_body_into_template(self, template_text: str, body_content: str, new_title: str) -> str:
#         template_text = re.sub(r'(\\title\{)[^}]*\}', r'\1' + new_title + r'}', template_text)

#         begin_doc_match = re.search(r'(\\begin{document})', template_text)
#         if not begin_doc_match:
#             raise ValueError("Template is missing \\begin{document}.")

#         # Check if there's a \maketitle command after \begin{document}
#         maketitle_match = re.search(r'(\\maketitle)', template_text)
#         ending_match = re.search(r'(\\end{document})', template_text)
#         if not ending_match:
#             raise ValueError("Template is missing \\end{document}.")
#         ending = template_text[ending_match.start():]

#         if maketitle_match:
#             insertion_point = maketitle_match.end()
#             return template_text[:insertion_point] + "\n" + body_content + "\n" + ending
#         else:
#             preamble = template_text[:begin_doc_match.end()]
#             return preamble + "\n" + body_content + "\n" + ending

#     def _assemble_body(self) -> str:
#         section_order = [
#             "Abstract",
#             "Introduction",
#             "Background",
#             "Method",
#             "Experimental Setup",
#             "Results",
#             "Related Work",
#             "Conclusion"
#         ]
#         body = ""
#         for section in section_order:
#             content = self.generated_sections.get(section, "")
#             if content:
#                 cleaned_content = self.clean_latex_content(content)
#                 # body += f"\\section{{{section}}}\n\n{cleaned_content}\n\n"
#                 body += f"{cleaned_content}\n\n"

#         # this is the temporary solution
#         body += "\n\n\\bibliography{custom}"

#         return body


#     def _compile_latex(self, cwd: str, template: str, output_pdf_path: str, timeout: int) -> None:
#         print("GENERATING LATEX")

#         fname = "latex.tex"
#         if template == 'acl':
#             fname = "acl_latex.tex"
#             cwd = osp.join(cwd, "latex")
#         elif template == 'iclr':
#             fname = "iclr2025_conference.tex"

#         compile_target = fname
#         if not osp.exists(osp.join(cwd, compile_target)):
#             print(f"File {compile_target} not found in {cwd}.")
#             return

#         if not compile_target:
#             print("Error: No .tex file found to compile. Aborting.")
#             return

#         commands = [
#             ["pdflatex", "-interaction=nonstopmode", compile_target],
#             ["bibtex", compile_target.replace(".tex","")],
#             ["pdflatex", "-interaction=nonstopmode", compile_target],
#             ["pdflatex", "-interaction=nonstopmode", compile_target],
#         ]
#         for command in commands:
#             try:
#                 result = subprocess.run(
#                     command,
#                     cwd=cwd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True,
#                     timeout=timeout,
#                 )
#                 print("Standard Output:\n", result.stdout)
#                 print("Standard Error:\n", result.stderr)
#             except subprocess.TimeoutExpired:
#                 print(f"Latex timed out after {timeout} seconds")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running command {' '.join(command)}: {e}")
#         print("FINISHED GENERATING LATEX")
#         # The PDF name is the same as compile_target minus .tex, e.g. 'latex.pdf' or 'template.pdf'
#         pdf_name = compile_target.replace(".tex", ".pdf")
#         try:
#             shutil.move(osp.join(cwd, pdf_name), output_pdf_path)
#         except FileNotFoundError:
#             print("Failed to rename PDF.")

#     def add_watermark(self, original_pdf_path: str, watermark_text: str, output_pdf_path: str) -> None:

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             watermark_pdf_path = tmp_file.name

#         c = canvas.Canvas(watermark_pdf_path, pagesize=letter)
#         c.saveState()
#         c.translate(300, 400)
#         c.rotate(45)
#         c.setFillColor(Color(0.95, 0.95, 0.95))
#         c.setFont("Helvetica-Bold", 28)

#         max_chars_per_line = 30
#         lines = textwrap.wrap(watermark_text, width=max_chars_per_line)

#         line_height = 35
#         y_offset = 0
#         for line in lines:
#             c.drawCentredString(0, y_offset, line)
#             y_offset -= line_height
#         c.restoreState()
#         c.showPage()
#         c.save()

#         # Read the original PDF and the watermark PDF.
#         original_reader = PdfReader(original_pdf_path)
#         watermark_reader = PdfReader(watermark_pdf_path)
#         if len(watermark_reader.pages) == 0:
#             print("Warning: Watermark PDF is empty. No watermark will be applied.")
#             return
#         watermark_page = watermark_reader.pages[0]
#         writer = PdfWriter()

#         for orig_page in original_reader.pages:
#             # Create a new blank page with the same dimensions as the original
#             new_page = PageObject.create_blank_page(
#                 width=orig_page.mediabox.width,
#                 height=orig_page.mediabox.height
#             )

#             new_page.merge_page(watermark_page)
#             new_page.merge_page(orig_page)

#             writer.add_page(new_page)

#         with open(output_pdf_path, "wb") as out_f:
#             writer.write(out_f)
#         print(f"Watermarked PDF saved to: {output_pdf_path}")
#         os.remove(watermark_pdf_path)

#     def generate_latex(self,
#                        output_pdf_path: str,
#                        template: str,
#                        name: str,
#                        timeout: int = 30,
#                        num_error_corrections: int = 5,
#                     ) -> None:

#         if template is not None:
#             body_content = self._assemble_body()

#             script_dir = osp.dirname(__file__)
#             project_root = osp.abspath(osp.join(script_dir, ".."))
#             source_template_dir = osp.join(project_root, "tiny_scientist", f"{template}_latex")

#             if osp.isdir(source_template_dir):
#                 dest_template_dir = osp.join(self.base_dir, "latex")

#                 if osp.exists(dest_template_dir):
#                     shutil.rmtree(dest_template_dir)
#                 shutil.copytree(source_template_dir, dest_template_dir)

#             self.update_custom_bib(dest_template_dir, template)

#             main_tex_path = ''
#             if template == 'acl':
#                 main_tex_path = osp.join(dest_template_dir, "latex", "acl_latex.tex")
#             elif template == 'iclr':
#                 main_tex_path = osp.join(dest_template_dir, "iclr2025_conference.tex")

#             with open(main_tex_path, "r", encoding="utf-8") as f:
#                 template_text = f.read()

#             final_content = self.insert_body_into_template(template_text, body_content, name)

#             with open(main_tex_path, "w", encoding="utf-8") as f:
#                 f.write(final_content)

#         else:
#             # TODO: MAYBE SUPPORT NON TEMPLATE GENERATION
#             pass

#         with open(main_tex_path, "r") as f:
#             final_content = f.read()

#         self._compile_latex(dest_template_dir, template, output_pdf_path, timeout)
#         self.add_watermark(
#                             output_pdf_path,
#                             watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
#                             output_pdf_path=output_pdf_path
#                         )



