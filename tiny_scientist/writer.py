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
        citation: Any,
        s2_api_key: Optional[str] = None
    ):
        """Initialize the PaperWriter with model and configuration."""
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.citation = citation
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")
        self.generated_sections: Dict[str, str] = {}
        self.dest_dir = ''

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

        with open(os.path.join(folder_name, "experiment.py"), "r") as f:
            code = f.read()
        with open(os.path.join(folder_name, "baseline_results.txt"), "r") as f:
            baseline_result = f.read()
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

        self._refine_paper()
        self._set_template(template)
        
        self.generated_sections = self.citation.process_citations(self.generated_sections, template, self.dest_dir)

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
        print('FINISHED REFINING SECTIONS')

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.HTTPError,
        on_backoff=lambda details: print(
            f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
            f"calling function {details['target'].__name__} at {time.strftime('%X')}"
        )
    )

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

    def _set_template(self, template: str) -> None:
        if template is not None:
            script_dir = osp.dirname(__file__)
            project_root = osp.abspath(osp.join(script_dir, ".."))
            source_template_dir = osp.join(project_root, "tiny_scientist", f"{template}_latex")

            if osp.isdir(source_template_dir):
                dest_template_dir = osp.join(self.base_dir, "latex")

                if osp.exists(dest_template_dir):
                    shutil.rmtree(dest_template_dir)
                shutil.copytree(source_template_dir, dest_template_dir)
        
            self.dest_dir = dest_template_dir

    def generate_latex(self,
                       output_pdf_path: str,
                       template: str,
                       name: str,
                       timeout: int = 30,
                       num_error_corrections: int = 5,
                    ) -> None:

        if template is not None:

            if template == 'acl':
                main_tex_path = osp.join(self.dest_dir, "latex", "acl_latex.tex")
            elif template == 'iclr':
                main_tex_path = osp.join(self.dest_dir, "iclr2025_conference.tex")

            with open(main_tex_path, "r", encoding="utf-8") as f:
                template_text = f.read()

            body_content = self._assemble_body()
            final_content = self.insert_body_into_template(template_text, body_content, name)

            with open(main_tex_path, "w", encoding="utf-8") as f:
                f.write(final_content)

        else:
            full_draft = self._assemble_full_draft()
            main_tex_path = osp.join(self.base_dir, "latex.tex")

            with open(main_tex_path, "w", encoding="utf-8") as f:
                f.write(full_draft)
            print("Generated full LaTeX draft.")

            self._check_latex_figures(main_tex_path)
            self._fix_latex_errors(main_tex_path, num_error_corrections)

        with open(main_tex_path, "r") as f:
            final_content = f.read()

        self._compile_latex(self.dest_dir, template, output_pdf_path, timeout)
        self.add_watermark(
                            output_pdf_path,
                            watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
                            output_pdf_path=output_pdf_path
                        )


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
