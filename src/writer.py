import json
import os
import os.path as osp
import re
import shutil
import subprocess
import time
from typing import Optional, Tuple, List, Dict

import backoff
import requests
import yaml

from .llm import (
    get_response_from_llm,
    extract_json_between_markers,
)


class PaperWriter:
    def __init__(
        self,
        model: str,
        client: any,
        base_dir: str,
        coder: any,
        s2_api_key: Optional[str] = None
    ):
        """Initialize the PaperWriter with model and configuration."""
        self.model = model
        self.client = client
        self.base_dir = base_dir
        self.coder = coder
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")
        
        # Load prompts
        yaml_path = os.path.join(os.path.dirname(__file__), "writer.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def perform_writeup(
        self,
        idea: Dict,
        num_cite_rounds: int = 20,
        engine: str = "semanticscholar"
    ) -> None:
        """Perform complete paper writeup process."""
        # Write initial sections
        self._write_abstract()
        
        # Write main sections
        for section in [
            "Introduction",
            "Background",
            "Method",
            "Experimental Setup",
            "Results",
            "Conclusion",
        ]:
            self._write_section(section)
        
        # Handle related work section
        self._write_related_work()
        
        # Add citations
        self._add_citations(num_cite_rounds, engine)
        
        # Perform second refinement
        self._refine_paper()
        
        # Generate final PDF
        self.generate_latex(f"{self.base_dir}/{idea['Name']}.pdf")

    def _write_abstract(self) -> None:
        """Write the abstract section."""
        abstract_prompt = self.prompts["abstract_prompt"].format(
            abstract_tips=self.prompts["section_tips"]["Abstract"]
        )
        self.coder.run(abstract_prompt)
        self._refine_section("Abstract")

    def _write_section(self, section: str) -> None:
        """Write a main section of the paper."""
        section_prompt = self.prompts["section_prompt"].format(
            section=section,
            section_tips=self.prompts["section_tips"][section]
        )
        self.coder.run(section_prompt)
        self._refine_section(section)

    def _write_related_work(self) -> None:
        """Write the related work section."""
        related_work_prompt = self.prompts["related_work_prompt"].format(
            related_work_tips=self.prompts["section_tips"]["Related Work"]
        )
        self.coder.run(related_work_prompt)

    def _refine_section(self, section: str) -> None:
        """Refine a section of the paper."""
        refinement_prompt = self.prompts["refinement_prompt"].format(
            section=section,
            error_list=self.prompts["error_list"]
        ).replace(r"{{", "{").replace(r"}}", "}")
        self.coder.run(refinement_prompt)

    def _refine_paper(self) -> None:
        """Perform second refinement of the entire paper."""
        self.coder.run(self.prompts["title_refinement_prompt"])
        
        for section in [
            "Abstract",
            "Related Work",
            "Introduction",
            "Background",
            "Method",
            "Experimental Setup",
            "Results",
            "Conclusion",
        ]:
            second_refinement_prompt = self.prompts["second_refinement_prompt"].format(
                section=section,
                tips=self.prompts["section_tips"][section],
                error_list=self.prompts["error_list"]
            ).replace(r"{{", "{").replace(r"}}", "}")
            self.coder.run(second_refinement_prompt)

    def _add_citations(self, num_cite_rounds: int, engine: str) -> None:
        """Add citations to the paper."""
        for i in range(num_cite_rounds):
            with open(osp.join(self.base_dir, "latex", "template.tex"), "r") as f:
                draft = f.read()
                
            prompt, done = self._get_citation_prompt(
                draft, i + 1, num_cite_rounds, engine
            )
            
            if done:
                break
                
            if prompt is not None:
                # Extract bibtex and update draft
                bibtex_string = prompt.split('"""')[1]
                search_str = r"\end{filecontents}"
                draft = draft.replace(search_str, f"{bibtex_string}{search_str}")
                
                # Save updated draft
                with open(osp.join(self.base_dir, "latex", "template.tex"), "w") as f:
                    f.write(draft)
                    
                self.coder.run(prompt)

    def _get_citation_prompt(
        self,
        draft: str,
        current_round: int,
        total_rounds: int,
        engine: str
    ) -> Tuple[Optional[str], bool]:
        """Get prompt for adding citations."""
        msg_history = []
        
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
                system_message=self.prompts["citation_system_prompt"].format(
                    total_rounds=total_rounds
                ),
                msg_history=msg_history,
            )
            
            if "No more citations needed" in text:
                print("No more citations needed.")
                return None, True

            json_output = extract_json_between_markers(text)
            if not json_output:
                return None, False
                
            query = json_output["Query"]
            papers = self._search_for_papers(query, engine=engine)
            
            if not papers:
                print("No papers found.")
                return None, False

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
                system_message=self.prompts["citation_system_prompt"].format(
                    total_rounds=total_rounds
                ),
                msg_history=msg_history,
            )
            
            if "Do not add any" in text:
                print("Do not add any.")
                return None, False

            json_output = extract_json_between_markers(text)
            if not json_output:
                return None, False
                
            desc = json_output["Description"]
            selected_papers = json_output["Selected"]
            
            if selected_papers == "[]":
                return None, False

            # Get bibtex entries for selected papers
            selected_indices = list(map(int, selected_papers.strip("[]").split(",")))
            if not all(0 <= i < len(papers) for i in selected_indices):
                return None, False
                
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]
            bibtex_string = "\n".join(bibtexs)

            # Format final prompt
            return self.prompts["citation_aider_format"].format(
                bibtex=bibtex_string,
                description=desc
            ), False

        except Exception as e:
            print(f"Error in citation generation: {e}")
            return None, False

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
    ) -> Optional[List[Dict]]:
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
    ) -> Optional[List[Dict]]:
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
    ) -> Optional[List[Dict]]:
        """Search papers using OpenAlex API."""
        import pyalex
        from pyalex import Works
        
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
    def _format_paper_results(papers: Optional[List[Dict]]) -> str:
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

    def generate_latex(
        self,
        output_pdf_path: str,
        timeout: int = 30,
        num_error_corrections: int = 5
    ) -> None:
        """Generate LaTeX PDF output."""
        cwd = osp.join(self.base_dir, "latex")
        template_path = osp.join(cwd, "template.tex")

        # Check citations
        self._check_latex_citations(template_path)
        
        # Check figures
        self._check_latex_figures(template_path)
        
        # Fix LaTeX errors
        self._fix_latex_errors(template_path, num_error_corrections)
        
        # Compile document
        self._compile_latex(cwd, output_pdf_path, timeout)

    def _check_latex_citations(self, template_path: str) -> None:
        """Check all references are valid and in the references.bib file."""
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
        
        for cite in cites:
            if cite not in bib_text:
                print(f"Reference {cite} not found in references.")
                prompt = f"""Reference {cite} not found in references.bib. Is this included under a different name?
If so, please modify the citation in template.tex to match the name in references.bib at the top. Otherwise, remove the cite."""
                self.coder.run(prompt)

    def _check_latex_figures(self, template_path: str) -> None:
        """Check all included figures are in the directory and not duplicated."""
        with open(template_path, "r") as f:
            tex_text = f.read()
            
        # Check figure existence
        referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
        all_figs = [f for f in os.listdir(self.base_dir) if f.endswith(".png")]
        
        for figure in referenced_figs:
            if figure not in all_figs:
                print(f"Figure {figure} not found in directory.")
                prompt = f"""The image {figure} not found in the directory. The images in the directory are: {all_figs}.
Please ensure that the figure is in the directory and that the filename is correct. Check the notes to see what each figure contains."""
                self.coder.run(prompt)

        # Check for duplicates
        duplicates = {x for x in referenced_figs if referenced_figs.count(x) > 1}
        if duplicates:
            for dup in duplicates:
                print(f"Duplicate figure found: {dup}.")
                prompt = f"""Duplicate figures found: {dup}. Ensure any figure is only included once.
If duplicated, identify the best location for the figure and remove any other."""
                self.coder.run(prompt)

        # Check for duplicate section headers
        sections = re.findall(r"\\section{([^}]*)}", tex_text)
        duplicates = {x for x in sections if sections.count(x) > 1}
        if duplicates:
            for dup in duplicates:
                print(f"Duplicate section header found: {dup}")
                prompt = f"""Duplicate section header found: {dup}. Ensure any section header is declared once.
If duplicated, identify the best location for the section header and remove any other."""
                self.coder.run(prompt)

    def _fix_latex_errors(
        self,
        template_path: str,
        num_error_corrections: int
    ) -> None:
        """Iteratively fix LaTeX errors."""
        for i in range(num_error_corrections):
            check_output = os.popen(
                f"chktex {template_path} -q -n2 -n24 -n13 -n1"
            ).read()
            
            if not check_output:
                break
                
            prompt = f"""Please fix the following LaTeX errors in `template.tex` guided by the output of `chktek`:
{check_output}.

Make the minimal fix required and do not remove or change any packages.
Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end."""
            self.coder.run(prompt)

    def _compile_latex(
        self,
        cwd: str,
        output_pdf_path: str,
        timeout: int
    ) -> None:
        """Compile LaTeX document."""
        print("GENERATING LATEX")

        commands = [
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["bibtex", "template"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
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

        # Move PDF to output location
        try:
            shutil.move(osp.join(cwd, "template.pdf"), output_pdf_path)
        except FileNotFoundError:
            print("Failed to rename PDF.")