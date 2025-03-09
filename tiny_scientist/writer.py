import os
import os.path as osp
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
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

from .llm import extract_json_between_markers, get_response_from_llm


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
        self.generate_latex(f"{self.base_dir}/{name}.pdf", template, name)

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

        # self._refine_section("Abstract")
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

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
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

    def update_custom_bib(self, dest_template_dir: str, template: str) -> None:
        all_keys = set()
        citation_patterns = [
            r"\\cite\{([^}]+)\}",
            r"\\citep\{([^}]+)\}",
            r"\\citet\{([^}]+)\}"
        ]

        for content in self.generated_sections.values():
            for pattern in citation_patterns:
                matches = re.findall(pattern, content)
                for m in matches:
                    keys = [key.strip() for key in m.split(",")]
                    all_keys.update(keys)

        if template == 'acl':
            bib_path = osp.join(dest_template_dir, "latex", "custom.bib")
        if template == 'iclr':
            # you should create a custom.bib file in the iclr folder
            bib_path = osp.join(dest_template_dir, "custom.bib")

        if osp.exists(bib_path):
            with open(bib_path, "r", encoding="utf-8") as f:
                bib_content = f.read()
            existing_keys = set(re.findall(r"@.+?\{([^,]+),", bib_content))
        else:
            bib_content = ""
            existing_keys = set()

        # 3. Find missing keys
        missing_keys = all_keys - existing_keys
        if not missing_keys:
            print("All citation keys are already present in custom.bib.")
            return

        # 4. For each missing key, get the bibtex entry
        new_entries = []
        for key in missing_keys:
            bibtex_entry = self._get_bibtex_for_key(key)
            if bibtex_entry:
                new_entries.append(bibtex_entry)
            else:
                print(f"Warning: Could not retrieve bibtex for key '{key}'.")

        # 5. Append the new entries to custom.bib
        if new_entries:
            updated_bib = bib_content + "\n" + "\n".join(new_entries)
            with open(bib_path, "w", encoding="utf-8") as f:
                f.write(updated_bib)
            print(f"Updated custom.bib with entries for: {', '.join(missing_keys)}")
        else:
            print("No new bibtex entries were added.")

    def _get_bibtex_for_key(self, key: str) -> Optional[str]:
        prompt = f"Provide the bibtex entry for the paper with citation key '{key}'. Output only the bibtex entry."
        bibtex_entry, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message="You are an expert in academic citations. Please provide a valid bibtex entry."
        )
        # A simple check: ensure it contains an @ and the key appears in the entry.
        if "@" in bibtex_entry and key in bibtex_entry:
            return bibtex_entry.strip()
        else:
            return None

    def _add_citations(self, num_cite_rounds: int, engine: str) -> None:
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
                # TODO: Handle TOO MANY REQUESTS ERROR
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
            print(f"Error in citation generation: {e}")
            return None, None, False

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

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.HTTPError,
        on_backoff=lambda details: print(
            f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
            f"calling function {details['target'].__name__} at {time.strftime('%X')}"
        )
    )

    def _search_for_papers(
        self,
        query: str,
        result_limit: int = 10,
        engine: str = "semanticscholar"
    ) -> Optional[List[Dict[str, Any]]]:
        """Search for papers using specified search engine."""
        if not query:
            return None

        if engine == "semanticscholar":
            return self._search_semanticscholar(query, result_limit)
        elif engine == "openalex":
            return self._search_openalex(query, result_limit)
        else:
            raise NotImplementedError(f"{engine=} not supported!")

    def _search_semanticscholar(
        self,
        query: str,
        result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Search papers using Semantic Scholar API."""
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()

        results = rsp.json()
        total = results["total"]
        time.sleep(1.0)

        return results["data"] if total else None

    def _search_openalex(
        self,
        query: str,
        result_limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Search papers using OpenAlex API."""

        mail = os.environ.get("OPENALEX_MAIL_ADDRESS")
        if mail:
            pyalex.config.email = mail
        else:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")

        works = Works().search(query).get(per_page=result_limit)
        if not works:
            return None

        papers = []
        for work in works:
            venue = "Unknown"
            for location in work["locations"]:
                if location["source"] is not None:
                    potential_venue = location["source"]["display_name"]
                    if potential_venue:
                        venue = potential_venue
                        break

            authors_list = [
                author["author"]["display_name"]
                for author in work["authorships"]
            ]
            authors = (
                " and ".join(authors_list)
                if len(authors_list) < 20
                else f"{authors_list[0]} et al."
            )

            abstract = work["abstract"] or ""
            if len(abstract) > 1000:
                print(
                    f"[WARNING] {work['title']}: Abstract length {len(abstract)} is too long! "
                    f"Using first 1000 chars."
                )
                abstract = abstract[:1000]

            papers.append({
                "title": work["title"],
                "authors": authors,
                "venue": venue,
                "year": work["publication_year"],
                "abstract": abstract,
                "citationCount": work["cited_by_count"],
            })

        return papers

    @staticmethod
    def _format_paper_results(papers: Optional[List[Dict[str, Any]]]) -> str:
        """Format paper results into a string."""
        if not papers:
            return "No papers found."

        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"{i}: {paper['title']}. {paper['authors']}. {paper['venue']}, {paper['year']}.\n"
                f"Abstract: {paper['abstract']}"
            )
        return "\n\n".join(paper_strings)

    def clean_latex_content(self, content: str) -> str:
        match = re.search(r'```latex\s*(.*?)\s*```', content, flags=re.DOTALL)
        if match:
            return match.group(1)

        # If no code block is found, perform minimal cleaning:
        lines = content.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Remove lines that are exactly code fences (```), but keep inline backticks if any.
            if stripped in ["```"]:
                continue
            # Remove markdown header lines (starting with '#' and not a LaTeX comment)
            if stripped.startswith("#") and not stripped.startswith("%"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def insert_body_into_template(self, template_text: str, body_content: str, new_title: str) -> str:
        template_text = re.sub(r'(\\title\{)[^}]*\}', r'\1' + new_title + r'}', template_text)

        begin_doc_match = re.search(r'(\\begin{document})', template_text)
        if not begin_doc_match:
            raise ValueError("Template is missing \\begin{document}.")

        # Check if there's a \maketitle command after \begin{document}
        maketitle_match = re.search(r'(\\maketitle)', template_text)
        ending_match = re.search(r'(\\end{document})', template_text)
        if not ending_match:
            raise ValueError("Template is missing \\end{document}.")
        ending = template_text[ending_match.start():]

        if maketitle_match:
            insertion_point = maketitle_match.end()
            return template_text[:insertion_point] + "\n" + body_content + "\n" + ending
        else:
            preamble = template_text[:begin_doc_match.end()]
            return preamble + "\n" + body_content + "\n" + ending

    def ensure_required_tags(self, content: str) -> str:
        match = re.search(r'(\\documentclass)', content)

        if match:
            content = content[match.start():]
        else:
            # If \documentclass is missing, prepend a default preamble.
            content = (
                "\\documentclass{article}\n"
                "\\usepackage{graphicx}\n"
                "\\usepackage{amsmath}\n"
                "\\usepackage{cite}\n"
                "\\usepackage{booktabs}\n"
                "\\usepackage{natbib}\n"
            ) + content

        if "\\begin{document}" not in content:
            # Insert \begin{document} after the preamble
            content = content.replace("\\documentclass{article}", "\\documentclass{article}\n\\begin{document}", 1)

        if "\\end{document}" not in content:
            content += "\n\\end{document}\n"
        return content

    def _assemble_body(self) -> str:
        section_order = [
            "Abstract",
            "Introduction",
            "Background",
            "Method",
            "Experimental Setup",
            "Results",
            "Related Work",
            "Conclusion"
        ]
        body = ""
        for section in section_order:
            content = self.generated_sections.get(section, "")
            if content:
                cleaned_content = self.clean_latex_content(content)
                # body += f"\\section{{{section}}}\n\n{cleaned_content}\n\n"
                body += f"{cleaned_content}\n\n"

        # this is the temporary solution
        body += "\n\n\\bibliography{custom}"

        return body

    def _assemble_full_draft(self) -> str:
        """
        Assemble the full LaTeX draft from generated sections.
        Clean each section's content to remove markdown artifacts and ensure the final document
        has all necessary LaTeX tags.
        """
        section_order = [
            "Abstract",
            "Introduction",
            "Background",
            "Method",
            "Experimental Setup",
            "Results",
            "Related Work",
            "Conclusion",
            "Citations"
        ]

        # Initial preamble including necessary packages
        preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{cite}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{natbib}\n"
            "\\begin{document}\n"
        )

        body = ""
        for section in section_order:
            content = self.generated_sections.get(section, "")
            if content:
                cleaned_content = self.clean_latex_content(content)
                body += f"\\section{{{section}}}\n\n{cleaned_content}\n\n"

        ending = "\\end{document}\n"
        full_draft = preamble + body + ending
        full_draft = self.ensure_required_tags(full_draft)
        return full_draft

    def _compile_latex(self, cwd: str, template: str, output_pdf_path: str, timeout: int) -> None:
        print("GENERATING LATEX")

        fname = "latex.tex"
        if template == 'acl':
            fname = "acl_latex.tex"
            cwd = osp.join(cwd, "latex")
        elif template == 'iclr':
            fname = "iclr2025_conference.tex"

        compile_target = fname
        if not osp.exists(osp.join(cwd, compile_target)):
            print(f"File {compile_target} not found in {cwd}.")
            return

        if not compile_target:
            print("Error: No .tex file found to compile. Aborting.")
            return

        commands = [
            ["pdflatex", "-interaction=nonstopmode", compile_target],
            ["bibtex", compile_target.replace(".tex","")],
            ["pdflatex", "-interaction=nonstopmode", compile_target],
            ["pdflatex", "-interaction=nonstopmode", compile_target],
        ]
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                print("Standard Output:\n", result.stdout)
                print("Standard Error:\n", result.stderr)
            except subprocess.TimeoutExpired:
                print(f"Latex timed out after {timeout} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error running command {' '.join(command)}: {e}")
        print("FINISHED GENERATING LATEX")
        # The PDF name is the same as compile_target minus .tex, e.g. 'latex.pdf' or 'template.pdf'
        pdf_name = compile_target.replace(".tex", ".pdf")
        try:
            shutil.move(osp.join(cwd, pdf_name), output_pdf_path)
        except FileNotFoundError:
            print("Failed to rename PDF.")

    def add_watermark(self, original_pdf_path: str, watermark_text: str, output_pdf_path: str) -> None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            watermark_pdf_path = tmp_file.name

        c = canvas.Canvas(watermark_pdf_path, pagesize=letter)
        c.saveState()
        c.translate(300, 400)
        c.rotate(45)
        c.setFillColor(Color(0.95, 0.95, 0.95))
        c.setFont("Helvetica-Bold", 28)

        max_chars_per_line = 30
        lines = textwrap.wrap(watermark_text, width=max_chars_per_line)

        line_height = 35
        y_offset = 0
        for line in lines:
            c.drawCentredString(0, y_offset, line)
            y_offset -= line_height
        c.restoreState()
        c.showPage()
        c.save()

        # Read the original PDF and the watermark PDF.
        original_reader = PdfReader(original_pdf_path)
        watermark_reader = PdfReader(watermark_pdf_path)
        if len(watermark_reader.pages) == 0:
            print("Warning: Watermark PDF is empty. No watermark will be applied.")
            return
        watermark_page = watermark_reader.pages[0]
        writer = PdfWriter()

        for orig_page in original_reader.pages:
            # Create a new blank page with the same dimensions as the original
            new_page = PageObject.create_blank_page(
                width=orig_page.mediabox.width,
                height=orig_page.mediabox.height
            )

            new_page.merge_page(watermark_page)
            new_page.merge_page(orig_page)

            writer.add_page(new_page)

        with open(output_pdf_path, "wb") as out_f:
            writer.write(out_f)
        print(f"Watermarked PDF saved to: {output_pdf_path}")
        os.remove(watermark_pdf_path)

    def generate_latex(self,
                       output_pdf_path: str,
                       template: str,
                       name: str,
                       timeout: int = 30,
                       num_error_corrections: int = 5,
                    ) -> None:

        if template is not None:
            body_content = self._assemble_body()

            script_dir = osp.dirname(__file__)
            project_root = osp.abspath(osp.join(script_dir, ".."))
            source_template_dir = osp.join(project_root, "tiny_scientist", f"{template}_latex")

            if osp.isdir(source_template_dir):
                dest_template_dir = osp.join(self.base_dir, "latex")

                if osp.exists(dest_template_dir):
                    shutil.rmtree(dest_template_dir)
                shutil.copytree(source_template_dir, dest_template_dir)

            self.update_custom_bib(dest_template_dir, template)

            main_tex_path = ''
            if template == 'acl':
                main_tex_path = osp.join(dest_template_dir, "latex", "acl_latex.tex")
            elif template == 'iclr':
                main_tex_path = osp.join(dest_template_dir, "iclr2025_conference.tex")

            with open(main_tex_path, "r", encoding="utf-8") as f:
                template_text = f.read()

            final_content = self.insert_body_into_template(template_text, body_content, name)

            with open(main_tex_path, "w", encoding="utf-8") as f:
                f.write(final_content)

        else:
            full_draft = self._assemble_full_draft()
            main_tex_path = osp.join(self.base_dir, "latex.tex")

            with open(main_tex_path, "w", encoding="utf-8") as f:
                f.write(full_draft)
            print("Generated full LaTeX draft.")

            self._check_latex_citations(main_tex_path)
            self._check_latex_figures(main_tex_path)
            self._fix_latex_errors(main_tex_path, num_error_corrections)

        with open(main_tex_path, "r") as f:
            final_content = f.read()

        self._compile_latex(dest_template_dir, template, output_pdf_path, timeout)
        self.add_watermark(
                            output_pdf_path,
                            watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
                            output_pdf_path=output_pdf_path
                        )

    def _check_latex_citations(self, template_path: str) -> None:
        """Check all references are valid and in the references.bib file, then fix them using LLM suggestions."""
        with open(template_path, "r") as f:
            tex_text = f.read()

        cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
        references_bib = re.search(
            r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
            tex_text,
            re.DOTALL,
        )

        if not references_bib:
            print("No references.bib found in template.tex")
            return

        bib_text = references_bib.group(1)
        cites = [cite.strip() for item in cites for cite in item.split(",")]

        updated = False
        for cite in cites:
            if cite not in bib_text:
                print(f"Reference {cite} not found in references.")
                prompt = (f"Reference {cite} not found in references.bib. Is this included under a different name? "
                        f"If so, please modify the citation in template.tex to match the name in references.bib at the top. Otherwise, remove the cite.")
                # Use get_response_from_llm to get a fix suggestion
                fix_response, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts["citation_system_prompt"].format(total_rounds=1)
                )
                print("Citation fix response:", fix_response)

                bib_text = bib_text.replace(cite, "")
                updated = True

        if updated:
            new_tex_text = re.sub(
                r"(\\begin{filecontents}{references.bib})(.*?)(\\end{filecontents})",
                r"\1" + bib_text + r"\3",
                tex_text,
                flags=re.DOTALL
            )
            with open(template_path, "w") as f:
                f.write(new_tex_text)
            print("Citations fixed and template updated.")

    def _check_latex_figures(self, template_path: str) -> None:
        """Check all included figures and fix issues using LLM suggestions."""
        with open(template_path, "r") as f:
            tex_text = f.read()

        referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
        all_figs = [f for f in os.listdir(self.base_dir) if f.endswith(".png")]

        updated = False
        for figure in referenced_figs:
            if figure not in all_figs:
                print(f"Figure {figure} not found in directory.")
                prompt = (f"The image {figure} is referenced in the LaTeX file but not found in the directory. "
                        f"The available images are: {all_figs}. Please suggest a correction (e.g., correct the filename or remove the reference).")
                fix_response, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts["write_system_prompt"]
                )
                print("Figure fix response:", fix_response)
                tex_text = tex_text.replace(f"\\includegraphics{{{figure}}}", "")
                updated = True

        # Optionally, handle duplicate figures or section headers similarly...
        if updated:
            with open(template_path, "w") as f:
                f.write(tex_text)
            print("Figure issues fixed and template updated.")

    def _fix_latex_errors(self, template_path: str, num_error_corrections: int) -> None:
        """Iteratively fix LaTeX errors while preserving existing content."""
        for i in range(num_error_corrections):
            check_output = os.popen(f"chktex {template_path} -q -n2 -n24 -n13 -n1").read()
            if not check_output:
                print("No LaTeX errors detected by chktex.")
                break

            # Read the original file content
            with open(template_path, "r") as f:
                original_content = f.read()

            # Create a prompt that instructs the LLM to only apply minimal fixes while preserving the original content.
            prompt = f"""Below is the content of a LaTeX document that needs to be minimally fixed for formatting errors without altering its content:
                --------------------------------------------------
                {original_content}
                --------------------------------------------------
                chktex reported the following errors:
                {check_output}
                Please output the corrected LaTeX document with only the necessary formatting fixes (e.g., removing markdown code fences, correcting misplaced characters) and without changing the core content.
                Output only the corrected LaTeX document content, with no additional explanations.
                """

            fix_response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts["write_system_prompt"]
            )

            # Validate the fix_response: it should still contain essential LaTeX markers.
            if fix_response and "\\documentclass" in fix_response and "\\begin{document}" in fix_response:
                with open(template_path, "w") as f:
                    f.write(fix_response)
                print(f"LaTeX errors fixed in round {i+1}.")
            else:
                print(f"Fix response in round {i+1} did not seem valid; no changes applied.")
                break
