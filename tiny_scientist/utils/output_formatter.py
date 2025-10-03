import abc
import os
import os.path as osp
import platform
import re
import shutil
import subprocess
import sys
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

    def _ensure_pdflatex(self) -> None:
        if shutil.which("pdflatex") is not None:
            return
        system = platform.system()

        try:
            if system == "Darwin":
                subprocess.run(["brew", "install", "--cask", "mactex"], check=True)
                print("[System] Installed MacTeX via Homebrew.")
            elif system == "Linux":
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "texlive-full"], check=True
                )
                print("[System] Installed TeX Live via apt.")
            else:
                raise RuntimeError(
                    "Unsupported system for automatic pdflatex installation."
                )
        except Exception as e:
            print(f"[Error] Automatic pdflatex installation failed: {e}")
            sys.exit(1)

    def _adjust_table_width_in_latex(self, latex_text: str) -> str:
        tabular_pattern = re.compile(
            r"(\\begin{table}.*?\\centering.*?)(\\begin{tabular}.*?\\end{tabular})",
            re.DOTALL,
        )

        def wrap_tabular_resizebox(match: Match[str]) -> str:
            table_start = match.group(1)
            tabular_block = match.group(2)
            return (
                f"{table_start}\\resizebox{{\\columnwidth}}{{!}}{{%\n"
                f"{tabular_block}\n}}"
            )

        latex_text = tabular_pattern.sub(wrap_tabular_resizebox, latex_text)

        array_math_pattern = re.compile(
            r"(\\\[\s*)(\\begin{array}.*?\\end{array})(\s*\\\])",
            re.DOTALL,
        )

        def wrap_array_resizebox(match: Match[str]) -> str:
            open_math = match.group(1)
            array_block = match.group(2)
            close_math = match.group(3)
            return (
                f"{open_math}\\resizebox{{\\columnwidth}}{{!}}{{%\n"
                f"${array_block}$\n}}{close_math}"
            )

        latex_text = array_math_pattern.sub(wrap_array_resizebox, latex_text)

        return latex_text

    def _convert_markdown_to_latex(self, content: str) -> str:
        """Convert common Markdown syntax to LaTeX equivalents"""

        # STEP 1: Fix mixed Markdown+LaTeX (e.g., **text\textbf{more** -> \textbf{text more})
        # Remove cases where ** and \textbf{ are mixed together
        content = re.sub(
            r"\*\*([^\*]*?)\\textbf\{([^\}]*?)\*\*", r"\\textbf{\1\2}", content
        )
        content = re.sub(r"\\textbf\{([^\}]*?)\*\*", r"\\textbf{\1}", content)
        content = re.sub(r"\*\*([^\*]*?)\\textbf\{", r"\\textbf{\1", content)
        
        # STEP 2: **bold** -> \textbf{bold}
        def replace_bold(match):
            text = match.group(1)
            # Don't convert if it looks like it's already LaTeX or in math mode
            if "\\" in text or "$" in text:
                return match.group(0)
            return f"\\textbf{{{text}}}"

        # Match **text** but not inside $...$ or \[...\]
        content = re.sub(r"\*\*([^\*]+?)\*\*", replace_bold, content)
        
        # *italic* -> \textit{italic} (single asterisk)
        def replace_italic(match):
            text = match.group(1)
            if "\\" in text or "$" in text:
                return match.group(0)
            return f"\\textit{{{text}}}"

        # Match *text* but not ** (already handled) and not inside math
        # Use word boundaries to avoid matching math multiplication
        content = re.sub(
            r"(?<!\*)\*([^\*\s][^\*]*?[^\*\s])\*(?!\*)", replace_italic, content
        )
        
        # `code` -> \texttt{code} (but avoid if already in verbatim or math)
        def replace_code(match):
            text = match.group(1)
            if "\\" in text:
                return match.group(0)
            # Escape special LaTeX characters in code
            text = text.replace("_", "\\_").replace("#", "\\#").replace("%", "\\%")
            return f"\\texttt{{{text}}}"
        
        content = re.sub(r"`([^`]+?)`", replace_code, content)
        
        # Normalize algorithm commands (support both old and new style)
        # The 'algorithmic' package uses uppercase (\STATE, \FOR, \IF, etc.)
        # The 'algpseudocode' package uses mixed case (\State, \For, \If, etc.)
        # We keep uppercase as-is since we load the 'algorithmic' package
        # But also ensure common variants work

        return content

    def _clean_body_content(self, body_content: str) -> str:
        patterns_to_remove = [
            r"\\documentclass(?:\[[^\]]*\])?\{[^\}]+\}",  # matches \documentclass[...]{...}
            r"\\begin\{document\}",
            r"\\end\{document\}",
            r"\\maketitle",
            r"\\title\{.*?\}",  # matches \title{...}
        ]

        for pattern in patterns_to_remove:
            body_content = re.sub(pattern, "", body_content, flags=re.DOTALL)

        # Convert Markdown syntax to LaTeX
        body_content = self._convert_markdown_to_latex(body_content)

        return body_content.strip()

    def _wrap_tables_in_latex(self, content: str) -> str:
        def replacer(match: Match[str]) -> str:
            tabular_block = match.group(1)

            # Check if the tabular block is already inside a table environment
            if (
                "\\begin{table}" in content[: match.start()]
                and "\\end{table}" in content[match.end() :]
            ):
                return tabular_block

            # Use [!t] to force table to top of page, reducing text wrapping issues
            return (
                "\\begin{table}[!t]\n"
                "\\centering\n"
                "\\resizebox{\\linewidth}{!}{%\n"
                f"{tabular_block}\n"
                "}\n"
                "\\caption{}\n"
                "\\label{}\n"
                "\\end{table}\n\n"
                "\\vspace{0.5em}\n"  # Add some space after table
            )

        return re.sub(
            r"(\\begin{tabular}.*?\\end{tabular})", replacer, content, flags=re.DOTALL
        )

    def _clean_latex_content(self, content: str) -> str:
        match = re.search(r"```latex\s*(.*?)\s*```", content, flags=re.DOTALL)
        if not match:
            match = re.search(r"```\s*(.*?)\s*```", content, flags=re.DOTALL)

        if match:
            content = match.group(1)

        # Remove LaTeX document structure commands
        patterns_to_remove = [
            r"\\documentclass(?:\[[^\]]*\])?\{[^\}]+\}",
            r"\\usepackage(?:\[[^\]]*\])?\{[^\}]+\}",
            r"\\begin\{document\}",
            r"\\end\{document\}",
            r"\\maketitle",
            r"\\title\{.*?\}",
            r"\\author\{.*?\}",
            r"\\bibliographystyle\{[^\}]+\}",
            r"\\bibliography\{[^\}]+\}",
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, "", content, flags=re.DOTALL)

        # Clean line by line
        lines = content.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped in ["```", "```latex", "```tex"]:
                continue
            if stripped.startswith("#") and not stripped.startswith("%"):
                continue
            if stripped.startswith("**") and stripped.endswith("**"):
                continue

            cleaned_lines.append(line)

        # Rejoin and fix text flow issues
        content = "\n".join(cleaned_lines)
        content = self._fix_text_flow(content)

        return content.strip()

    def _fix_text_flow(self, content: str) -> str:
        """Fix common text flow and formatting issues"""

        def fix_broken_sentences(text: str) -> str:
            lines = text.split("\n")
            fixed_lines = []
            i = 0

            while i < len(lines):
                current_line = lines[i].strip()

                # Skip empty lines
                if not current_line:
                    fixed_lines.append(lines[i])
                    i += 1
                    continue

                # Check if we should merge with next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()

                    should_merge = (
                        current_line
                        and next_line
                        and len(current_line) > 20
                        and len(next_line) > 10
                        and not current_line.endswith((".", "!", "?", ":", ";"))
                        and not next_line.startswith(("\\", "-", "•", "*"))
                        and not next_line[0].isupper()
                        and not current_line.endswith("\\\\")
                        and not re.match(r"^\d+\.", next_line)  # Not numbered list
                    )

                    if should_merge:
                        # Merge the lines with a space
                        merged_line = current_line + " " + next_line
                        fixed_lines.append(merged_line)
                        i += 2  # Skip next line since we merged it
                        continue

                fixed_lines.append(lines[i])
                i += 1

            return "\n".join(fixed_lines)

        # Apply the fix
        content = fix_broken_sentences(content)

        # Fix multiple spaces
        content = re.sub(r" +", " ", content)

        # Fix spacing around punctuation
        content = re.sub(r" +([,.;:!?])", r"\1", content)

        # Fix paragraph spacing (ensure proper spacing between paragraphs)
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        # Fix spacing around section headers
        content = re.sub(r"\n(\\section\{[^}]+\})\n*", r"\n\n\1\n", content)
        content = re.sub(r"\n(\\subsection\{[^}]+\})\n*", r"\n\n\1\n", content)

        return content

    def _assemble_body(self, contents: Dict[str, str]) -> str:
        """Enhanced body assembly with better text flow"""
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

        body_parts = []

        for section in section_order:
            content = contents.get(section, "")

            if content:
                # Clean the content
                cleaned_content = self._clean_latex_content(content)

                # Wrap tables
                cleaned_content = self._wrap_tables_in_latex(cleaned_content)

                # Add section title if needed
                section_title = section_titles[section]
                if section_title is not None:
                    # Check if section title is already present
                    starts_with_section = re.match(
                        rf"\\section\{{{re.escape(section_title)}\}}",
                        cleaned_content,
                        re.IGNORECASE,
                    )
                    if not starts_with_section:
                        cleaned_content = (
                            f"\\section{{{section_title}}}\n\n{cleaned_content}"
                        )

                body_parts.append(cleaned_content)

        # Join all parts with proper spacing
        body = "\n\n".join(body_parts)

        # Apply table width adjustments
        body = self._adjust_table_width_in_latex(body)

        # Add bibliography
        body += "\n\n\\bibliography{custom}"

        # Final cleanup
        body = self._final_cleanup(body)

        return body

    def _final_cleanup(self, content: str) -> str:
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        content = re.sub(r"\n(\\begin\{[^}]+\})", r"\n\n\1", content)
        content = re.sub(r"(\\end\{[^}]+\})\n", r"\1\n\n", content)

        content = re.sub(r"(\n\\section\{[^}]+\})\n*", r"\1\n\n", content)
        content = re.sub(r"(\n\\subsection\{[^}]+\})\n*", r"\1\n\n", content)

        content = re.sub(
            r"(\n\\begin\{itemize\}|\n\\begin\{enumerate\})", r"\1\n", content
        )
        content = re.sub(r"(\n\\end\{itemize\}|\n\\end\{enumerate\})", r"\1\n", content)

        content = re.sub(r"(\w+)\s*\n\s*(\\cite\{[^}]+\})", r"\1~\2", content)
        content = re.sub(r"(\$[^$]+\$)\s*\n\s*(\w)", r"\1 \2", content)

        return content.strip()

    def _insert_body_into_template(
        self, template_text: str, body_content: str, new_title: str
    ) -> str:
        template_text = re.sub(
            r"(\\title\{)[^}]*\}", r"\1" + new_title + r"}", template_text
        )

        begin_doc_match = re.search(r"(\\begin{document})", template_text)
        if not begin_doc_match:
            raise ValueError("Template is missing \\begin{document}.")

        # Inject essential math and algorithm packages before \begin{document}
        math_packages = (
            "\n% Essential math packages (auto-injected by tiny_scientist)\n"
            "\\usepackage{amsmath}   % For advanced math environments and \\text{}\n"
            "\\usepackage{amssymb}   % For additional math symbols\n"
            "\\usepackage{amsthm}    % For theorem environments\n"
            "\\usepackage{mathtools} % Enhanced math support\n"
            "\\usepackage{bm}        % For bold math symbols\n\n"
            "% Algorithm packages (supporting both old and new syntax)\n"
            "\\usepackage{algorithm}      % For algorithm floating environment\n"
            "\\usepackage{algorithmic}    % Old-style commands (\\STATE, \\FOR, etc.)\n"
            "\\usepackage{algpseudocode}  % New-style commands (\\State, \\For, etc.)\n"
            "\\usepackage{algorithmicx}   % Enhanced algorithm support\n\n"
        )

        # Check if amsmath is already present (avoid duplicate injection)
        if "amsmath" not in template_text:
            # Inject before \begin{document}
            template_text = (
                template_text[: begin_doc_match.start()]
                + math_packages
                + template_text[begin_doc_match.start() :]
            )
            # Update match after injection
            begin_doc_match = re.search(r"(\\begin{document})", template_text)
            print(
                "[INFO] Injected essential packages: math (amsmath, amssymb, amsthm, mathtools, bm) + algorithms (algorithm, algorithmic, algpseudocode, algorithmicx)"
            )

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

    def _clean_invalid_citations(
        self, content: str, dest_template_dir: str, template: str
    ) -> str:
        if template == "acl":
            bib_path = osp.join(dest_template_dir, "custom.bib")
        if template == "iclr":
            bib_path = osp.join(dest_template_dir, "custom.bib")

        with open(bib_path, "r") as f:
            bib_content = f.read()
        
        valid_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content, re.IGNORECASE))
        
        valid_keys = {k.strip() for k in valid_keys}

        print(f"[DEBUG] Found {len(valid_keys)} valid bibtex keys in custom.bib")
        if valid_keys:
            print(f"[DEBUG] Sample keys: {list(valid_keys)[:5]}")

        def citation_replacer(match: Match[str]) -> str:
            cite_cmd = match.group(1)  # e.g., 'cite', 'citep', 'citet'
            raw_keys = match.group(2)
            # Clean each key: remove extra braces and whitespace
            keys = []
            for k in raw_keys.split(","):
                cleaned = k.strip().lstrip("{").rstrip("}").strip()
                if cleaned:
                    keys.append(cleaned)

            # Filter valid keys
            valid = [k for k in keys if k in valid_keys]
            if valid:
                return f"\\{cite_cmd}{{{','.join(valid)}}}"
            else:
                print(f"[WARNING] Removing invalid citation keys from \\{cite_cmd}: {keys}")
                return ""

        # Match all citation commands: \cite{}, \citep{}, \citet{}, \citealp{}, \citealt{}, etc.
        return re.sub(r"\\(cite[a-z]*)\{+([^\}]+)\}+", citation_replacer, content)


class TemplateDownloader:
    @staticmethod
    def download_acl_template(output_dir: str) -> str:
        print(f"Downloading ACL template from GitHub to {output_dir}")
        dest_template_dir = osp.join(output_dir, "latex")
        os.makedirs(dest_template_dir, exist_ok=True)

        # GitHub repository URL for ACL
        acl_api_url = "https://api.github.com/repos/acl-org/acl-style-files/contents/"
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
    def __init__(self, model: str, client: Any, latex_fix_prompt: Any = None) -> None:
        self.template = "acl"
        self.model = model
        self.client = client
        self.bib_manager = BibManager(model, client)
        self.watermarker = WaterMarker()
        self.latex_fix_prompt = latex_fix_prompt

    def run(
        self,
        content: Dict[str, str],
        references: Dict[str, Any],
        output_dir: str,
        output_pdf_path: str,
        name: str,
        timeout: int = 30,
    ) -> None:
        body_content = self._assemble_body(content)
        body_content = self._clean_body_content(body_content)
        dest_template_dir = TemplateDownloader.download_acl_template(output_dir)

        self.bib_manager._update_bib_cite(references, dest_template_dir, self.template)

        body_content = self._clean_invalid_citations(
            body_content, dest_template_dir, self.template
        )
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
        try:
            self.watermarker._add_watermark(
                output_pdf_path,
                watermark_text="THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
                output_pdf_path=output_pdf_path,
            )
            print("[INFO] Watermark added successfully")
        except Exception as e:
            print(f"[WARNING] Failed to add watermark: {e}")
            print("[INFO] Continuing without watermark to avoid PDF corruption")

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

    def _analyze_latex_log(self, log_path: str) -> Dict[str, Any]:
        """Analyze LaTeX log file for fatal errors"""
        if not osp.exists(log_path):
            return {"has_errors": False, "errors": []}
        
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
        
        errors = []
        
        # Check for fatal errors
        fatal_patterns = [
            (r"! Undefined control sequence\.\n.*\n.*\\(\w+)", "undefined_command"),
            (r"! LaTeX Error: (.*)", "latex_error"),
            (r"! LaTeX Warning: (.*)", "latex_warning"),
            (r"! Package (\w+) Error: (.*)", "package_error"),
            (r"Runaway argument\?", "runaway_argument"),
            (r"! File ended while scanning use of (\\.*)\.", "unclosed_environment"),
            (r"! Missing \\begin\{document\}", "missing_begin_document"),
        ]
        
        for pattern, error_type in fatal_patterns:
            matches = re.finditer(pattern, log_content, re.MULTILINE)
            for match in matches:
                errors.append({
                    "type": error_type,
                    "message": match.group(0),
                    "details": match.groups() if match.groups() else ()
                })
        
        return {
            "has_errors": len(errors) > 0,
            "errors": errors,
            "log_content": log_content
        }
    
    def _fix_latex_errors(self, tex_path: str, log_analysis: Dict[str, Any]) -> bool:
        """Use LLM to automatically fix LaTeX errors"""
        if not log_analysis["has_errors"]:
            return False
        
        if not self.latex_fix_prompt:
            print("[LaTeX Fix] No fix prompt available, skipping LLM fix")
            return False
        
        with open(tex_path, "r", encoding="utf-8") as f:
            tex_content = f.read()
        
        # Prepare error summary
        error_summary = []
        for i, error in enumerate(log_analysis["errors"][:10], 1):  # Limit to 10 errors
            error_summary.append(f"{i}. {error['type']}: {error['message'][:200]}")
        error_summary_str = "\n".join(error_summary)
        
        # Get last 100 lines of log for context
        log_lines = log_analysis["log_content"].split("\n")
        log_excerpt = "\n".join(log_lines[-100:])
        
        try:
            # Import get_response_from_llm here to avoid circular imports
            from ..llm import get_response_from_llm
            
            print(f"[LaTeX Fix] Calling LLM to fix {len(log_analysis['errors'])} errors...")
            
            # Format the fix prompt
            fix_prompt = self.latex_fix_prompt.latex_fix_prompt.format(
                log_excerpt=log_excerpt,
                error_summary=error_summary_str,
                tex_content=tex_content
            )
            
            # Call LLM to fix the LaTeX
            fixed_content, _ = get_response_from_llm(
                msg=fix_prompt,
                client=self.client,
                model=self.model,
                system_message=self.latex_fix_prompt.latex_fix_system_prompt,
                print_debug=False,
                temperature=0.1,  # Low temperature for consistent fixes
            )
            
            # Clean up the response (remove markdown code blocks if present)
            fixed_content = fixed_content.strip()
            if fixed_content.startswith("```"):
                # Remove markdown code blocks
                lines = fixed_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                fixed_content = "\n".join(lines)
            
            # Write the fixed content
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            
            print("[LaTeX Fix] LLM successfully applied fixes to LaTeX file")
            return True
            
        except Exception as e:
            print(f"[LaTeX Fix] LLM fix failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _compile_latex(self, cwd: str, output_pdf_path: str, timeout: int) -> None:
        """LaTeX compilation using pdflatex with automatic error fixing"""
        self._ensure_pdflatex()

        fname = "acl_latex.tex"
        tex_path = osp.join(cwd, fname)
        log_path = osp.join(cwd, fname.replace(".tex", ".log"))
        
        if not osp.exists(tex_path):
            print(f"File {fname} not found in {cwd}.")
            return

        max_fix_attempts = 2
        for attempt in range(max_fix_attempts):
            try:
                # Step 1: First pdflatex run (generates .aux file)
                result1 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-file-line-error", fname],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )

                # Step 2: Run bibtex (generates .bbl file from .aux and .bib)
                base_name = fname.replace(".tex", "")
                subprocess.run(
                    ["bibtex", base_name],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )

                # Step 3: Second pdflatex run (reads .bbl file and updates references)
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-file-line-error", fname],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )

                # Step 4: Third pdflatex run (resolves all cross-references and citations)
                result4 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-file-line-error", fname],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                
                # Analyze log for fatal errors
                log_analysis = self._analyze_latex_log(log_path)
                
                if log_analysis["has_errors"] and attempt < max_fix_attempts - 1:
                    print(f"[LaTeX] Detected {len(log_analysis['errors'])} errors, attempting automatic fix...")
                    for err in log_analysis["errors"][:3]:  # Show first 3 errors
                        print(f"  - {err['type']}: {err['message'][:100]}")
                    
                    # Try to fix errors
                    if self._fix_latex_errors(tex_path, log_analysis):
                        print(f"[LaTeX] Retrying compilation (attempt {attempt + 2}/{max_fix_attempts})...")
                        continue
                    else:
                        print("[LaTeX] No automatic fixes available, continuing...")
                        break
                else:
                    # Success or max attempts reached
                    if log_analysis["has_errors"]:
                        print(f"[LaTeX] Compilation completed with {len(log_analysis['errors'])} errors")
                        print(f"[LaTeX] Check {log_path} for details")
                    break

            except subprocess.TimeoutExpired:
                print(f"LaTeX compilation timed out after {timeout} seconds.")
                return
            except FileNotFoundError:
                print(
                    "LaTeX commands not found. Make sure pdflatex and bibtex are installed."
                )
                return
            except Exception as e:
                print(f"[LaTeX] Compilation error: {e}")
                return

        # Move the PDF to final location
        pdf_source = osp.join(cwd, fname.replace(".tex", ".pdf"))
        if osp.exists(pdf_source):
            try:
                shutil.move(pdf_source, output_pdf_path)
                print(f"[LaTeX] PDF successfully generated: {output_pdf_path}")
            except Exception as e:
                print(f"[LaTeX] Failed to move PDF: {e}")


class ICLROutputFormatter(BaseOutputFormatter):
    def __init__(self, model: str, client: Any, latex_fix_prompt: Any = None) -> None:
        self.template = "iclr"
        self.model = model
        self.client = client
        self.bib_manager = BibManager(model, client)
        self.watermarker = WaterMarker()
        self.latex_fix_prompt = latex_fix_prompt

    def run(
        self,
        content: Dict[str, str],
        references: Dict[str, Any],
        output_dir: str,
        output_pdf_path: str,
        name: str,
        timeout: int = 30,
    ) -> None:
        body_content = self._assemble_body(content)
        body_content = self._clean_body_content(body_content)
        dest_template_dir = TemplateDownloader.download_iclr_template(output_dir)

        self.bib_manager._update_bib_cite(references, dest_template_dir, self.template)

        body_content = self._clean_invalid_citations(
            body_content, dest_template_dir, self.template
        )
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
        try:
            self.watermarker._add_watermark(
                output_pdf_path,
                watermark_text="THIS PAPER WAS AUTONOMOUSLY GENERATED BY THE TINY_SCIENTIST",
                output_pdf_path=output_pdf_path,
            )
            print("[INFO] Watermark added successfully")
        except Exception as e:
            print(f"[WARNING] Failed to add watermark: {e}")
            print("[INFO] Continuing without watermark to avoid PDF corruption")

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
        self._ensure_pdflatex()

        fname = "iclr2025_conference.tex"
        if not osp.exists(osp.join(cwd, fname)):
            print(f"File {fname} not found in {cwd}.")
            return

        try:
            # Method 1: Try latexmk (should handle everything automatically)
            result = subprocess.run(
                [
                    "latexmk",
                    "-pdf",
                    "-bibtex",
                    "-interaction=nonstopmode",
                    "-file-line-error",
                    fname,
                ],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )

            pdf_name = fname.replace(".tex", ".pdf")
            pdf_path = osp.join(cwd, pdf_name)

            # If latexmk succeeded and PDF exists, we're done
            if osp.exists(pdf_path):
                stdout_output = result.stdout.decode("utf-8", errors="ignore")
                if not (
                    "undefined" in stdout_output.lower()
                    and "citation" in stdout_output.lower()
                ):
                    shutil.move(pdf_path, output_pdf_path)
                    return

            # Method 2: Manual compilation (only if latexmk failed or has citation issues)
            # Step 1: First pdflatex run
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-file-line-error", fname],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )

            # Step 2: Run bibtex (only if .aux file exists)
            aux_file = osp.join(cwd, fname.replace(".tex", ".aux"))
            if osp.exists(aux_file):
                base_name = fname.replace(".tex", "")
                subprocess.run(
                    ["bibtex", base_name],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )

            # Step 3: Second pdflatex run (to read .bbl file and update references)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-file-line-error", fname],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )

            # Step 4: Third pdflatex run (to resolve all cross-references and citations)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-file-line-error", fname],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired:
            print(f"LaTeX compilation timed out after {timeout} seconds.")
            return
        except FileNotFoundError:
            print(
                "LaTeX commands not found. Make sure pdflatex and bibtex are installed."
            )
            return
        except Exception:
            return

        # Move the PDF
        pdf_source = osp.join(cwd, fname.replace(".tex", ".pdf"))
        if osp.exists(pdf_source):
            try:
                shutil.move(pdf_source, output_pdf_path)
            except Exception:
                pass
