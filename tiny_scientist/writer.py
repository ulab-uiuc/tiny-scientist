import json
import os
import os.path as osp
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cairosvg
from rich import print

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
    ) -> None:
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.template = template
        self.temperature = temperature
        self.searcher: BaseTool = PaperSearchTool()
        self.drawer: BaseTool = DrawerTool(model, prompt_template_dir, temperature)
        self.formatter: BaseOutputFormatter
        self.config = Config(prompt_template_dir)
        if self.template == "acl":
            self.formatter = ACLOutputFormatter(model=self.model, client=self.client)
        elif self.template == "iclr":
            self.formatter = ICLROutputFormatter(model=self.model, client=self.client)

        self.prompts = self.config.prompt_template.writer_prompt

    def run(self, idea: Dict[str, Any], experiment_dir: str) -> Tuple[str, str]:
        # Read experiment log instead of code
        log_path = osp.join(experiment_dir, "experiment_logger.log")
        experiment_log = ""
        if osp.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    # Read the log content, potentially summarizing or extracting key parts later
                    experiment_log = f.read()
                print(f"[INFO] Read experiment log from: {log_path}")
            except Exception as e:
                print(f"[WARNING] Failed to read experiment log: {e}")
                experiment_log = "Experiment log could not be read."
        else:
            print(f"[WARNING] Experiment log not found at: {log_path}")
            experiment_log = "Experiment log not found."

        # Read experiment results (this part remains the same)
        results_path = osp.join(experiment_dir, "experiment_results.txt")
        experiment_result = ""
        if osp.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    experiment_result = f.read()
            except Exception as e:
                print(f"[WARNING] Failed to read experiment results: {e}")
                experiment_result = "Experiment results could not be read."
        else:
            print(f"[WARNING] Experiment results not found at: {results_path}")
            experiment_result = "Experiment results not found."

        # Read baseline results (this part remains the same)
        baseline_path = osp.join(experiment_dir, "baseline_results.txt")
        baseline_result = ""
        if osp.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline_result = f.read()
        else:
            baseline_result = ""

        self.generated_sections: Dict[str, Any] = {}
        self.references: Dict[str, Any] = {}

        self._write_abstract(idea)

        for section in [
            "Method",
            "Experimental_Setup",
            "Results",
            "Introduction",
            "Discussion",
            "Conclusion",
        ]:
            # Pass the experiment log instead of code to relevant sections
            self._write_section(idea, experiment_log, experiment_result, section, baseline_result)

        self._write_related_work(idea)
        self._refine_paper()
        self._add_citations(idea)

        paper_name = idea.get("Title", "Research Paper")

        output_pdf_path = f"{self.output_dir}/{paper_name}.pdf"
        self.formatter.run(
            content=self.generated_sections,
            references=self.references,
            output_dir=self.output_dir,
            output_pdf_path=output_pdf_path,
            name=paper_name,
        )
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
            system_message=self.prompts.write_system_prompt,
        )

        self.generated_sections["Abstract"] = abstract_content

    def _generate_diagram_for_section(
        self, section: str, content: str
    ) -> Optional[Dict[str, str]]:
        """Generate a diagram for a specific section if appropriate."""
        if section in ["Introduction", "Method", "Experimental_Setup", "Results"]:
            try:
                # Use the section text content to generate the diagram
                diagram_input = self.formatter.strip_latex(content) # Get plain text
                diagram_result = self.drawer.run(diagram_input)
                if diagram_result and "diagram" in diagram_result:
                    diagram = diagram_result["diagram"]
                    if not diagram.get("svg"): # Check if SVG is actually generated
                        print(f"[INFO] No SVG content returned for {section} diagram.")
                        return None

                    # Ensure latex output directory exists
                    latex_dir = os.path.join(self.output_dir, "latex")
                    os.makedirs(latex_dir, exist_ok=True)

                    pdf_filename = f"diagram_{section.lower()}.pdf"
                    pdf_path = os.path.join(latex_dir, pdf_filename)

                    try:
                        cairosvg.svg2pdf(
                            bytestring=diagram["svg"].encode("utf-8"), write_to=pdf_path
                        )
                        print(f"[INFO] Saved diagram to {pdf_path}")
                    except Exception as svg_err:
                        print(f"[ERROR] Failed to convert SVG to PDF for {section}: {svg_err}")
                        print(f"SVG content was:\n{diagram['svg'][:500]}...") # Log partial SVG for debugging
                        return None # Don't include broken diagram

                    # Sanitize caption for LaTeX
                    caption = diagram.get("summary", "Generated diagram for " + section)
                    caption = caption.replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace("&", "\\&").replace("#", "\\#").replace("^", "\\^").replace("~", "\\~").replace("%", "\\%")

                    return {
                        "summary": caption,
                        "tex": f"""
        \\begin{{figure}}[htbp] % Use htbp for better placement
        \\centering
        \\includegraphics[width=0.9\\linewidth]{{{pdf_filename}}}
        \\caption{{{caption}}}
        \\label{{fig:{section.lower()}_diagram}} % Add a label
        \\end{{figure}}
        """,
                    }
            except Exception as e:
                print(f"[WARNING] Failed to generate diagram for {section}: {e}")
                traceback.print_exc()
        return None

    def _write_section(
        self,
        idea: Dict[str, Any],
        experiment_log: str, # Changed from code to experiment_log
        experiment_result: str,
        section: str,
        baseline_result: Optional[str] = "",
    ) -> None:
        title = idea.get("Title", "Research Paper")
        experiment_description = idea.get("Experiment") # Keep original experiment description
        print(f"Writing section: {section}...")

        # Prepare context for the prompt
        # For sections needing method/setup details, provide the log
        method_context = experiment_log if section in ["Method", "Experimental_Setup", "Results", "Discussion"] else ""
        # Summarize log if too long? Maybe later optimization.
        # For now, just pass it.
        if len(method_context) > 8000: # Arbitrary limit to avoid excessive prompt length
            print(f"[INFO] Experiment log for {section} is long ({len(method_context)} chars), truncating.")
            method_context = method_context[:4000] + "... [LOG TRUNCATED] ..." + method_context[-4000:]

        if section in ["Introduction"]:
            # Introduction might benefit from knowing the outcome (results) and the goal (idea)
            method_summary = self.formatter.strip_latex(
                self.generated_sections.get("Method", "Method section not yet generated.")
            )
            abstract_content = self.formatter.strip_latex(
                self.generated_sections.get("Abstract", "Abstract not yet generated.")
            )
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                title=title,
                problem=idea["Problem"],
                importance=idea["Importance"],
                difficulty=idea["Difficulty"],
                novelty=idea["NoveltyComparison"],
                experiment=experiment_description, # Use original description here
                method_section=method_summary, # Provide generated method summary
                abstract=abstract_content,
                # Optionally add experiment_results summary here if needed
                experiment_results_summary=experiment_result[:1000] + ("..." if len(experiment_result)>1000 else ""),
            )
        elif section in ["Conclusion"]:
            # Conclusion summarizes the work based on results and initial goal
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                experiment=experiment_description, # Use original description
                # Provide key results for conclusion
                experiment_results_summary=experiment_result[:1000] + ("..." if len(experiment_result)>1000 else ""),
                 # Optionally provide a summary of the method section
                method_summary=self.formatter.strip_latex(self.generated_sections.get("Method", ""))[:1000] + "...",
            )
        elif section in ["Method", "Experimental_Setup"]:
            # These sections describe *how* the experiment was done, using the log
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                problem=idea["Problem"],
                importance=idea["Importance"],
                difficulty=idea["Difficulty"],
                novelty=idea["NoveltyComparison"],
                experiment=experiment_description, # Original description for context
                experiment_log=method_context, # Add experiment log
            )
        elif section in ["Results", "Discussion"]:
            # These sections interpret the results, potentially referencing the log for *how* they were obtained
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                experiment=experiment_description, # Original description for context
                baseline_results=baseline_result,
                experiment_results=experiment_result, # Full results
                experiment_log=method_context, # Add experiment log for context
            )
        else:
             # Default case or other sections
             # Check if the prompt template expects experiment_log or code
             # For safety, provide both if unsure, or adjust prompts
             try:
                 section_prompt = self.prompts.section_prompt[section].format(
                     section_tips=self.prompts.section_tips[section],
                     experiment=experiment_description,
                     # Add other fields expected by the specific prompt template
                     title=title,
                     problem=idea["Problem"],
                     novelty=idea["NoveltyComparison"],
                     # Provide log and results for context if relevant
                     experiment_log=method_context,
                     experiment_results=experiment_result,
                     baseline_results=baseline_result,
                 )
             except KeyError:
                 print(f"[ERROR] Prompt key missing for section '{section}'. Skipping.")
                 return

        section_content, _ = get_response_from_llm(
            msg=section_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt,
        )

        # Generate diagram for appropriate sections based on the generated content
        diagram = self._generate_diagram_for_section(section, section_content)

        if diagram:
            section_content += f"\n\n{diagram['tex']}"

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
            )

            new_titles = None
            try:
                # Try parsing as direct JSON list first
                potential_json = response.strip()
                if potential_json.startswith("[") and potential_json.endswith("]"):
                    new_titles = json.loads(potential_json)
                else:
                    # Fallback to extracting from markers
                    new_titles = extract_json_between_markers(response)
                    # If extracted data is a dict with a key like 'papers', get the list
                    if isinstance(new_titles, dict) and len(new_titles) == 1:
                        key = list(new_titles.keys())[0]
                        if isinstance(new_titles[key], list):
                           new_titles = new_titles[key]

            except json.JSONDecodeError as e:
                 print(f"[WARNING] JSONDecodeError in citation round {round_num+1}: {e}. Response: {response[:200]}...")
                 # Try simple regex for lines starting with "-" or numbered lists
                 lines = response.split('\n')
                 found_titles = [re.sub(r"^[ \-\d\.\*]+ ", "", line).strip() for line in lines if re.match(r"^[ \-\d\.\*]+ *\w", line)]
                 if found_titles:
                     print(f"[INFO] Extracted titles using regex fallback: {found_titles}")
                     new_titles = found_titles
                 else:
                    new_titles = None # Ensure it's None if parsing fails

            if new_titles and isinstance(new_titles, list):
                # Clean titles (remove potential numbering/bullets again)
                cleaned_titles = [re.sub(r"^[ \-\d\.\*]+ ", "", str(t)).strip().strip('"') for t in new_titles if str(t).strip()]
                collected_papers.extend(cleaned_titles)
                collected_papers = list(dict.fromkeys(collected_papers)) # Remove duplicates
                print(f"Round {round_num+1}: Collected {len(cleaned_titles)} new titles.")
            else:
                print(f"Round {round_num+1}: No valid titles returned or extracted.")
                # Optional: Add a retry mechanism here if needed

            if len(collected_papers) >= total_num_papers:
                break
            time.sleep(1) # Respect potential API rate limits

        print(f"Total unique related work titles collected: {len(collected_papers)}")
        return collected_papers[:total_num_papers] # Return up to the requested number

    def _search_reference(self, paper_list: List[str]) -> Dict[str, Any]:
        results_dict = {}
        if not paper_list:
            return results_dict

        print(f"Searching references for {len(paper_list)} titles...")
        processed_count = 0
        for paper_name in paper_list:
            if not paper_name or not isinstance(paper_name, str): # Skip invalid entries
                continue
            try:
                # Use PaperSearchTool
                result = self.searcher.run(paper_name)

                if result:
                    # Find the best matching key (case-insensitive comparison might help)
                    found_match = False
                    for key in result:
                        # Simple exact match first (or normalized match)
                        if key.lower() == paper_name.lower():
                             results_dict[key] = result[key] # Use the key from the result
                             found_match = True
                             break
                    # If no exact match, take the first result as fallback
                    if not found_match:
                        first_key = next(iter(result))
                        results_dict[first_key] = result[first_key]
                        print(f"[INFO] No exact match for '{paper_name}', using first result: '{first_key}'")

                processed_count += 1
                if processed_count % 5 == 0:
                     print(f"Searched {processed_count}/{len(paper_list)} references...")

                time.sleep(1.0) # Rate limiting
            except Exception as e:
                print(f"[ERROR] While searching reference for '{paper_name}': {e}")
                # traceback.print_exc() # Can be noisy

        print(f"Reference search complete. Found details for {len(results_dict)} titles.")
        return results_dict

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        print("Writing section: Related Work...")
        citations = self._get_citations_related_work(
            idea, num_cite_rounds=2, total_num_papers=10
        )

        paper_source = self._search_reference(citations)
        self.references = paper_source # Store found references

        if not paper_source:
            print("[WARNING] No references found for Related Work section.")
            self.generated_sections["Related_Work"] = "No related work could be automatically retrieved for this topic." # Add placeholder
            return

        # Prepare reference list for the prompt (Title: Bibtex Key (Year))
        reference_prompt_list = []
        bibtex_map = {}
        for title, meta in paper_source.items():
            bibtex = meta.get("bibtex", "")
            match = re.search(r"@\w+\{(.+?),", bibtex)
            bibtex_key = match.group(1) if match else title.lower().replace(" ", "_")[:20]
            year_match = re.search(r"year\s*=\s*\{?(\d{4})\}?", bibtex, re.IGNORECASE)
            year = year_match.group(1) if year_match else "N/A"
            reference_prompt_list.append(f"- {title} (Ref Key: {bibtex_key}, Year: {year})")
            bibtex_map[title] = bibtex_key # Map original title to bibtex key

        reference_list_str = "\n".join(reference_prompt_list)
        # Escape for format string - already handled by f-string usually, but be careful
        # reference_list_str = reference_list_str.replace("{\", "{{\").replace("}\", "}}")

        experiment_description = idea.get("Experiment", "No experiment details provided")

        related_work_prompt = self.prompts.related_work_prompt.format(
            related_work_tips=self.prompts.section_tips["Related_Work"],
            experiment=experiment_description,
            references=reference_list_str, # Pass the formatted list with keys
        )

        relatedwork_content, _ = get_response_from_llm(
            msg=related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt_related_work,
        )

        # Replace placeholder citations like \\cite{Title...} or \\cite{Ref Key: ...}
        # with actual \\cite{bibtex_key}
        def replace_citation(match): # Use a function for replacement logic
            placeholder = match.group(1).strip()
            # Try matching by Ref Key first
            key_match = re.match(r"Ref Key: (\w+)", placeholder, re.IGNORECASE)
            if key_match and key_match.group(1) in bibtex_map.values():
                return f"\\\\cite{{{key_match.group(1)}}}\""
            # Try matching by title (more fragile)
            found_key = None
            for title, key in bibtex_map.items():
                 # Basic substring match (can be improved with fuzzy matching)
                if placeholder.lower() in title.lower() or title.lower() in placeholder.lower():
                    found_key = key
                    break
            if found_key:
                return f"\\\\cite{{{found_key}}}\""
            else:
                # If no match found, keep the original placeholder or a warning
                print(f"[WARNING] Could not map citation placeholder '{placeholder}' to a bibtex key.")
                return f"\\\\cite{{{placeholder}}} % FIXME: Unmapped citation"

        relatedwork_content = re.sub(r"\\cite\{(.+?)\}\"", replace_citation, relatedwork_content)

        self.generated_sections["Related_Work"] = relatedwork_content

    def _refine_section(self, section: str) -> None:
        """Refine a section of the paper."""
        # This function seems less used now, consider integrating into _refine_paper
        if section not in self.generated_sections:
            print(f"[INFO] Skipping refinement for non-existent section: {section}")
            return

        print(f"Refining section (old method): {section}...")
        refinement_prompt = (
            self.prompts.refinement_prompt.format(
                section=section,
                section_content=self.generated_sections[section],
                error_list=self.prompts.error_list,
            )
            # .replace(r"{{", "{") # No longer needed with f-string?
            # .replace(r"}}", "}")
        )

        refined_section, _ = get_response_from_llm(
            msg=refinement_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt,
        )

        self.generated_sections[section] = refined_section

    def _refine_paper(self) -> None:
        print("Refining paper sections...")
        # Combine sections into a draft for context
        # Exclude Title if it exists, handle potential diagram TeX
        draft_sections = []
        for section, content in self.generated_sections.items():
             if section != "Title":
                 # Basic cleaning of potential figure environments for the draft context
                 cleaned_content = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "[Diagram Placeholder]", content, flags=re.DOTALL)
                 draft_sections.append(f"\\section{{{section}}}\\n\\n{cleaned_content}")

        full_draft = "\n\n".join(draft_sections)
        if len(full_draft) > 15000: # Limit context size for refinement prompts
             print(f"[INFO] Full draft for refinement is long ({len(full_draft)} chars), truncating context.")
             full_draft = full_draft[:7500] + "\n... [DRAFT TRUNCATED] ...\n" + full_draft[-7500:]

        # Refine Title first
        try:
            print("Refining title...")
            title_refinement_prompt = self.prompts.title_refinement_prompt.format(full_draft=full_draft)
            refined_title, _ = get_response_from_llm(
                msg=title_refinement_prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.write_system_prompt,
            )
             # Basic cleaning of the title
            refined_title = refined_title.strip().strip('"').strip()
            # Remove potential markdown like **Title**
            refined_title = re.sub(r"\*\*(.*?)\*\*", r"\1", refined_title)
            if refined_title:
                 self.generated_sections["Title"] = refined_title
                 print(f"Refined Title: {refined_title}")
        except KeyError:
            print("[WARNING] title_refinement_prompt key not found in prompts.")
        except Exception as e:
            print(f"[ERROR] Failed to refine title: {e}")

        # Refine other sections using the full draft as context
        sections_to_refine = [
            "Abstract",
            "Introduction",
            # "Background", # Add if you have this section
            "Method",
            "Experimental_Setup",
            "Results",
            "Discussion", # Add Discussion refinement
            "Conclusion",
            "Related_Work", # Also refine related work for flow
        ]

        for section in sections_to_refine:
            if section in self.generated_sections:
                print(f"Refining section: {section}...")
                original_content = self.generated_sections[section]
                # Include diagram TeX in the content passed for refinement, but maybe not in full_draft context

                # Use second_refinement_prompt if available, otherwise skip?
                if "second_refinement_prompt" in self.prompts:
                    try:
                        # Ensure all keys are present in section_tips and prompts
                        tips = self.prompts.section_tips.get(section, "General writing tips.")
                        error_list = self.prompts.get("error_list", "- Vague statements\n- Grammatical errors")

                        second_refinement_prompt = (
                            self.prompts.second_refinement_prompt.format(
                                section=section,
                                tips=tips,
                                full_draft=full_draft, # Provide truncated draft as context
                                section_content=original_content, # Provide original section content
                                error_list=error_list,
                            )
                            # .replace(r"{{", "{") # No longer needed?
                            # .replace(r"}}", "}")
                        )

                        refined_section, _ = get_response_from_llm(
                            msg=second_refinement_prompt,
                            client=self.client,
                            model=self.model,
                            system_message=self.prompts.write_system_prompt, # Generic writing prompt
                            temperature=0.5 # Slightly lower temp for refinement
                        )

                        # Basic check if refinement didn't fail or return empty
                        if refined_section and len(refined_section) > 0.5 * len(original_content):
                            self.generated_sections[section] = refined_section
                        else:
                             print(f"[WARNING] Refinement for {section} produced suspiciously short/empty result. Keeping original.")

                    except KeyError as e:
                         print(f"[WARNING] Missing key '{e}' in prompt format for refining {section}. Skipping refinement.")
                    except Exception as e:
                         print(f"[ERROR] Failed to refine section {section}: {e}")
                         traceback.print_exc()
                else:
                    print(f"[INFO] second_refinement_prompt not found. Skipping refinement for {section}.")

    def _add_citations(self, idea: Dict[str, Any]) -> None:
        print("Adding citations to sections...")
        idea_title = idea.get("Title", "Research Paper")

        # Use existing references gathered during related work generation
        existing_references = self.references
        if not existing_references:
             print("[INFO] No existing references found to add citations from.")
             return

        # Prepare reference list for citation embedding prompts
        reference_prompt_list = []
        bibtex_map = {}
        title_to_key_map = {}
        for title, meta in existing_references.items():
            bibtex = meta.get("bibtex", "")
            match = re.search(r"@\w+\{(.+?),", bibtex)
            if match:
                bibtex_key = match.group(1)
                year_match = re.search(r"year\s*=\s*\{?(\d{4})\}?", bibtex, re.IGNORECASE)
                year = year_match.group(1) if year_match else "N/A"
                reference_prompt_list.append(f"- {title} (Ref Key: {bibtex_key}, Year: {year})")
                bibtex_map[title] = bibtex_key
                title_to_key_map[title.lower()] = bibtex_key # For easier lookup
            else:
                 print(f"[WARNING] Could not extract bibtex key for: {title}")

        reference_list_str = "\n".join(reference_prompt_list)

        # Sections where citations might be relevant
        sections_to_cite = ["Introduction", "Method", "Experimental_Setup", "Discussion", "Related_Work"]

        for section in sections_to_cite:
            if section in self.generated_sections:
                print(f"Attempting to add citations to: {section}...")
                try:
                    original_content = self.generated_sections[section]

                    # Use embed_citation_prompt to add citations based on existing refs
                    if "embed_citation_prompt" not in self.prompts:
                        print(f"[WARNING] embed_citation_prompt not found. Cannot add citations to {section}.")
                        continue

                    embed_citation_prompt = self.prompts.embed_citation_prompt.format(
                        section=section,
                        section_content=original_content,
                        references=reference_list_str, # Provide list of available references
                    )

                    # Use a model good at following instructions precisely
                    # System prompt might need to emphasize *only* adding citations from the list
                    cited_section, _ = get_response_from_llm(
                        msg=embed_citation_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts.citation_system_prompt, # Or a dedicated one for embedding
                        temperature=0.3 # Low temperature for precision
                    )

                    # Post-processing: Replace placeholders with actual \\cite{key}
                    def replace_embed_citation(match):
                        placeholder = match.group(1).strip()
                         # Try matching Ref Key first
                        key_match = re.search(r"Ref Key: (\w+)", placeholder, re.IGNORECASE)
                        if key_match and key_match.group(1) in bibtex_map.values():
                            return f"\\cite{{{key_match.group(1)}}}"
                        # Try matching title (case-insensitive)
                        found_key = title_to_key_map.get(placeholder.lower())
                        if found_key:
                            return f"\\cite{{{found_key}}}"

                        # Fallback: Check if placeholder *is* a bibtex key already
                        if placeholder in bibtex_map.values():
                            return f"\\cite{{{placeholder}}}"

                        # Last resort: maybe LLM used title directly
                        for title, key in bibtex_map.items():
                             if placeholder.lower() in title.lower() or title.lower() in placeholder.lower():
                                 print(f"[INFO] Citation heuristic match for '{placeholder}' -> '{key}'")
                                 return f"\\cite{{{key}}}"

                        print(f"[WARNING] Could not map citation placeholder '{placeholder}' in {section}.")
                        return f"\\cite{{{placeholder}}} % FIXME: Unmapped citation"

                    # Apply the replacement
                    final_cited_section = re.sub(r"\\cite\{(.+?)\}", replace_embed_citation, cited_section)

                    # Basic check: ensure content wasn't drastically changed or deleted
                    if len(final_cited_section) > 0.7 * len(original_content):
                         self.generated_sections[section] = final_cited_section
                         # Count citations added (simple check)
                         original_cites = len(re.findall(r"\\cite\{.*?\}\"", original_content))
                         new_cites = len(re.findall(r"\\cite\{.*?\}\"", final_cited_section))
                         print(f"[INFO] Citations in {section}: {original_cites} -> {new_cites}")
                    else:
                         print(f"[WARNING] Citation embedding for {section} significantly changed content length. Reverting.")

                except KeyError as e:
                    print(f"[WARNING] Missing key '{e}' in prompt format for adding citations to {section}.")
                except Exception as e:
                    print(f"[ERROR] Failed to add citations to section {section}: {e}")
                    traceback.print_exc()
