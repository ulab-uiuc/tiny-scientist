import abc
import os
import os.path as osp
import re
import shutil
import subprocess
from typing import Any, Dict, Match

import requests
from rich import print

from .bib_manager import BibManager
from .water_marker import WaterMarker


class BaseOutputFormatter(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        content: Dict[str, Any],
        references: Dict[str, Any],
        output_dir: str,
        output_pdf_path: str,
        name: str,
        timeout: int = 30,
    ) -> None:
        pass

    def _clean_latex_content(self, content: str) -> str:
        match = re.search(r"```latex\s*(.*?)\s*```", content, flags=re.DOTALL)
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

    def _wrap_tables_in_latex(self, content: str) -> str:
        def replacer(match: Match[str]) -> str:
            tabular_block = match.group(1)

            # Check if the tabular block is already inside a table environment
            if (
                "\\begin{table}" in content[: match.start()]
                and "\\end{table}" in content[match.end() :]
            ):
                return tabular_block  # Already inside a table, skip wrapping

            return (
                "\\begin{table}[ht]\n"
                "\\centering\n"
                "\\resizebox{\\linewidth}{!}{%\n"
                f"{tabular_block}\n"
                "}\n"
                "\\caption{}\n"
                "\\label{}\n"
                "\\end{table}"
            )

        return re.sub(
            r"(\\begin{tabular}.*?\\end{tabular})", replacer, content, flags=re.DOTALL
        )

    def _assemble_body(self, contents: Dict[str, Dict[str, Any]]) -> str:
        section_order = [
            "Abstract",
            "Introduction",
            "Related_Work",
            "Method",
            "Experimental_Setup",
            "Results",
            "Discussion",
            "Conclusion",
        ]

        section_titles = {
            "Abstract": None,
            "Introduction": "Introduction",
            "Related_Work": "Related Work",
            "Method": "Method",
            "Experimental_Setup": "Experimental Setup",
            "Results": "Results",
            "Discussion": "Discussion",
            "Conclusion": "Conclusion",
        }

        body = ""
        for section in section_order:
            raw = contents.get(section, "")
            content = raw.get("text", "") if isinstance(raw, dict) else raw
            if content:
                cleaned_content = self._clean_latex_content(content)
                cleaned_content = self._wrap_tables_in_latex(cleaned_content)
                section_title = section_titles[section]
                if section_title is not None:
                    starts_with_section = re.match(
                        rf"\\section\{{{re.escape(section_title)}\}}",
                        cleaned_content,
                        re.IGNORECASE,
                    )
                    starts_with_text = cleaned_content.lower().startswith(
                        section_title.lower()
                    )
                    if not starts_with_section and not starts_with_text:
                        body += f"\\section{{{section_title}}}\n"
                body += f"{cleaned_content}\n\n"

        body += "\n\n\\bibliography{custom}"
        return body

    def _insert_body_into_template(
        self, template_text: str, body_content: str, new_title: str
    ) -> str:
        template_text = re.sub(
            r"(\\title\{)[^}]*\}", r"\1" + new_title + r"}", template_text
        )

        begin_doc_match = re.search(r"(\\begin{document})", template_text)
        if not begin_doc_match:
            raise ValueError("Template is missing \\begin{document}.")

        # Check if there's a \maketitle command after \begin{document}
        maketitle_match = re.search(r"(\\maketitle)", template_text)
        ending_match = re.search(r"(\\end{document})", template_text)
        if not ending_match:
            raise ValueError("Template is missing \\end{document}.")
        ending = template_text[ending_match.start() :]

        if maketitle_match:
            insertion_point = maketitle_match.end()
            return template_text[:insertion_point] + "\n" + body_content + "\n" + ending
        else:
            preamble = template_text[: begin_doc_match.end()]
            return preamble + "\n" + body_content + "\n" + ending


class TemplateDownloader:
    @staticmethod
    def download_acl_template(output_dir: str) -> str:
        print(f"Downloading ACL template from GitHub to {output_dir}")
        dest_template_dir = osp.join(output_dir, "latex")
        os.makedirs(dest_template_dir, exist_ok=True)

        # GitHub repository URL for ACL
        acl_api_url = (
            "https://api.github.com/repos/acl-org/acl-style-files/contents/latex"
        )
        response = requests.get(acl_api_url)
        response.raise_for_status()

        files_data = response.json()
        for file_info in files_data:
            if file_info["type"] == "file":
                file_url = file_info["download_url"]
                filename = file_info["name"]

                print(f"Downloading {filename}...")
                file_response = requests.get(file_url)
                file_response.raise_for_status()

                with open(osp.join(dest_template_dir, filename), "wb") as f:
                    f.write(file_response.content)

        return dest_template_dir

    @staticmethod
    def download_iclr_template(output_dir: str) -> str:
        print(f"Downloading ICLR template from GitHub to {output_dir}")
        dest_template_dir = osp.join(output_dir, "latex")
        os.makedirs(dest_template_dir, exist_ok=True)

        # Get list of files in the iclr2025 directory
        iclr_api_url = (
            "https://api.github.com/repos/ICLR/Master-Template/contents/iclr2025"
        )
        response = requests.get(iclr_api_url)
        response.raise_for_status()

        files_data = response.json()

        # Download each file in the directory
        for file_info in files_data:
            if file_info["type"] == "file":
                file_url = file_info["download_url"]
                filename = file_info["name"]

                print(f"Downloading {filename}...")
                file_response = requests.get(file_url)
                file_response.raise_for_status()

                with open(osp.join(dest_template_dir, filename), "wb") as f:
                    f.write(file_response.content)

        return dest_template_dir


class ACLOutputFormatter(BaseOutputFormatter):
    def __init__(self, model: str, client: Any) -> None:
        self.template = "acl"
        self.bib_manager = BibManager(model, client)
        self.watermarker = WaterMarker()

    def run(
        self,
        content: Dict[str, Any],
        references: Dict[str, Any],
        output_dir: str,
        output_pdf_path: str,
        name: str,
        timeout: int = 30,
    ) -> None:
        body_content = self._assemble_body(content)
        dest_template_dir = TemplateDownloader.download_acl_template(output_dir)

        self.bib_manager._update_bib_cite(references, dest_template_dir, self.template)

        main_tex_path = osp.join(dest_template_dir, "acl_latex.tex")

        with open(main_tex_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        final_content = self._insert_body_into_template(
            template_text, body_content, name
        )

        with open(main_tex_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        with open(main_tex_path, "r") as f:
            final_content = f.read()

        self._compile_latex(dest_template_dir, output_pdf_path, timeout)
        self.watermarker._add_watermark(
            output_pdf_path,
            watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
            output_pdf_path=output_pdf_path,
        )

    def _set_output_dir(self, output_dir: str) -> str:
        script_dir = osp.dirname(__file__)
        project_root = osp.abspath(osp.join(script_dir, ".."))
        source_template_dir = osp.join(
            project_root, "tiny_scientist", f"{self.template}_latex"
        )

        if osp.isdir(source_template_dir):
            dest_template_dir = osp.join(output_dir, "latex")

            if osp.exists(dest_template_dir):
                shutil.rmtree(dest_template_dir)
            shutil.copytree(source_template_dir, dest_template_dir)

        return dest_template_dir

    def _compile_latex(self, cwd: str, output_pdf_path: str, timeout: int) -> None:
        fname = "acl_latex.tex"

        compile_target = fname
        if not osp.exists(osp.join(cwd, compile_target)):
            print(f"File {compile_target} not found in {cwd}.")
            return

        if not compile_target:
            print("Error: No .tex file found to compile. Aborting.")
            return

        commands = [
            ["pdflatex", "-interaction=nonstopmode", compile_target],
            ["bibtex", compile_target.replace(".tex", "")],
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


class ICLROutputFormatter(BaseOutputFormatter):
    def __init__(self, model: str, client: Any) -> None:
        self.template = "iclr"
        self.bib_manager = BibManager(model, client)
        self.watermarker = WaterMarker()

    def run(
        self,
        content: Dict[str, Any],
        references: Dict[str, Any],
        output_dir: str,
        output_pdf_path: str,
        name: str,
        timeout: int = 30,
    ) -> None:
        body_content = self._assemble_body(content)
        dest_template_dir = TemplateDownloader.download_iclr_template(output_dir)

        self.bib_manager._update_bib_cite(references, dest_template_dir, self.template)

        main_tex_path = osp.join(dest_template_dir, "iclr2025_conference.tex")

        with open(main_tex_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        final_content = self._insert_body_into_template(
            template_text, body_content, name
        )

        with open(main_tex_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        with open(main_tex_path, "r") as f:
            final_content = f.read()

        self._compile_latex(dest_template_dir, output_pdf_path, timeout)
        self.watermarker._add_watermark(
            output_pdf_path,
            watermark_text="CAUTION!!! THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
            output_pdf_path=output_pdf_path,
        )

    def _set_output_dir(self, output_dir: str) -> str:
        script_dir = osp.dirname(__file__)
        project_root = osp.abspath(osp.join(script_dir, ".."))
        source_template_dir = osp.join(
            project_root, "tiny_scientist", f"{self.template}_latex"
        )

        if osp.isdir(source_template_dir):
            dest_template_dir = osp.join(output_dir, "latex")

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
            ["bibtex", compile_target.replace(".tex", "")],
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
