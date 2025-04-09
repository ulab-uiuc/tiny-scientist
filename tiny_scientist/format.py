import abc
import os
import os.path as osp
import re
import shutil
import subprocess
import tempfile
import textwrap
from typing import Any, Dict, Optional

from pypdf import PageObject, PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .llm import get_response_from_llm


class BaseFormat(abc.ABC):
    @abc.abstractmethod
    def run(self,
            content: Dict[str, Any],
            references: Dict[str, Any],
            base_dir: str,
            output_pdf_path: str,
            name: str,
            timeout: int = 30
            ) -> None:
        pass

    def _clean_latex_content(self, content: str) -> str:
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

    def _assemble_body(self, contents: Dict[str, Dict[str, Any]]) -> str:
        section_order = [
            "Abstract",
            "Introduction",
            "Related Work",
            "Method",
            "Experimental Setup",
            "Results",
            "Discussion",
            "Conclusion"
        ]

        body = ""
        for section in section_order:
            raw = contents.get(section, "")
            content = raw.get("text", "") if isinstance(raw, dict) else raw
            if content:
                cleaned_content = self._clean_latex_content(content)
                body += f"{cleaned_content}\n\n"

        body += "\n\n\\bibliography{custom}"
        return body

    def _insert_body_into_template(self,
                                   template_text: str,
                                   body_content: str,
                                   new_title: str
                                   ) -> str:
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


class Watermark:
    def _add_watermark(self, original_pdf_path: str, watermark_text: str, output_pdf_path: str) -> None:

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

        original_reader = PdfReader(original_pdf_path)
        watermark_reader = PdfReader(watermark_pdf_path)
        if len(watermark_reader.pages) == 0:
            print("Warning: Watermark PDF is empty. No watermark will be applied.")
            return

        watermark_page = watermark_reader.pages[0]
        writer = PdfWriter()

        for orig_page in original_reader.pages:
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

class Bib_Manager:
    def __init__(self, model: str, client: Any) -> None:
        self.model = model
        self.client = client

    def _update_bib_cite(self, references: Dict[str, Any], dest_template_dir: str, template: str) -> None:

        if template == 'acl':
            bib_path = osp.join(dest_template_dir, "latex", "custom.bib")
        if template == 'iclr':
            # you should create a custom.bib file in the iclr folder
            bib_path = osp.join(dest_template_dir, "custom.bib")

        bib_entries = []
        for meta in references.values():
            bibtex = meta.get("bibtex", "").strip()
            if bibtex:
                bib_entries.append(bibtex)

        if not bib_entries:
            print("No BibTeX entries to write.")
            return

        # Write all entries to the bib file
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(bib_entries))

        print(f"custom.bib created with {len(bib_entries)} entries.")

    def _get_bibtex_for_key(self, key: str) -> Optional[str]:
        prompt = f"Provide the bibtex entry for the paper with citation key '{key}'. Output only the bibtex entry."
        try:
            result = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message="You are an expert in academic citations. Please provide a valid bibtex entry."
            )

            if isinstance(result, tuple):
                bibtex_entry = result[0]
            else:
                bibtex_entry = result

            if isinstance(bibtex_entry, str) and "@" in bibtex_entry and key in bibtex_entry:
                return bibtex_entry.strip()
            else:
                print(f"Invalid bibtex returned for key: {key}")
                return None

        except Exception as e:
            print(f"Error fetching bibtex for key '{key}': {e}")
            return None


class ACLFormat(BaseFormat):
    def __init__(self,
                 model: str,
                 client: Any
                ) -> None:
        self.template = "acl"
        self.bib_manager = Bib_Manager(model, client)
        self.watermark = Watermark()

    def run(self,
            content: Dict[str, Any],
            references: Dict[str, Any],
            base_dir: str,
            output_pdf_path: str,
            name: str,
            timeout: int = 30) -> None:

        body_content = self._assemble_body(content)
        dest_template_dir = self._set_output_dir(base_dir)

        self.bib_manager._update_bib_cite(references, dest_template_dir, self.template)

        main_tex_path = osp.join(dest_template_dir, "latex", "acl_latex.tex")

        with open(main_tex_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        final_content = self._insert_body_into_template(template_text, body_content, name)

        with open(main_tex_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        with open(main_tex_path, "r") as f:
            final_content = f.read()

        self._compile_latex(dest_template_dir, output_pdf_path, timeout)
        self.watermark._add_watermark(
                            output_pdf_path,
                            watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
                            output_pdf_path=output_pdf_path
                            )


    def _set_output_dir(self, base_dir: str) -> str:
        script_dir = osp.dirname(__file__)
        project_root = osp.abspath(osp.join(script_dir, ".."))
        source_template_dir = osp.join(project_root, "tiny_scientist", f"{self.template}_latex")

        if osp.isdir(source_template_dir):
            dest_template_dir = osp.join(base_dir, "latex")

            if osp.exists(dest_template_dir):
                shutil.rmtree(dest_template_dir)
            shutil.copytree(source_template_dir, dest_template_dir)

        return dest_template_dir

    def _compile_latex(self, cwd: str, output_pdf_path: str, timeout: int) -> None:
        fname = "acl_latex.tex"
        cwd = osp.join(cwd, "latex")

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


class ICLRFormat(BaseFormat):
    def __init__(self,
                 model: str,
                 client: Any
                ) -> None:
        self.template = "iclr"
        self.bib_manager = Bib_Manager(model, client)
        self.watermark = Watermark()

    def run(self,
            content: Dict[str, Any],
            references: Dict[str, Any],
            base_dir: str,
            output_pdf_path: str,
            name: str,
            timeout: int = 30) -> None:

        body_content = self._assemble_body(content)
        dest_template_dir = self._set_output_dir(base_dir)


        self.bib_manager._update_bib_cite(references, dest_template_dir, self.template)

        main_tex_path = osp.join(dest_template_dir, "iclr2025_conference.tex")

        with open(main_tex_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        final_content = self._insert_body_into_template(template_text, body_content, name)

        with open(main_tex_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        with open(main_tex_path, "r") as f:
            final_content = f.read()

        self._compile_latex(dest_template_dir, output_pdf_path, timeout)
        self.watermark._add_watermark(
                            output_pdf_path,
                            watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
                            output_pdf_path=output_pdf_path
                            )


    def _set_output_dir(self, base_dir: str) -> str:
        script_dir = osp.dirname(__file__)
        project_root = osp.abspath(osp.join(script_dir, ".."))
        source_template_dir = osp.join(project_root, "tiny_scientist", f"{self.template}_latex")

        if osp.isdir(source_template_dir):
            dest_template_dir = osp.join(base_dir, "latex")

            if osp.exists(dest_template_dir):
                shutil.rmtree(dest_template_dir)
            shutil.copytree(source_template_dir, dest_template_dir)

        return dest_template_dir

    def _compile_latex(self, cwd: str, output_pdf_path: str, timeout: int) -> None:
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
