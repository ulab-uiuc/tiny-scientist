import json
import os
import os.path as osp
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from rich import print

from .configs import Config
from .tool import BaseTool, PaperSearchTool # Removed DrawerTool
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

# Removed cairosvg import as DrawerTool is removed


class WriterMini: # Renamed class
    def __init__(
        self,
        model: str,
        output_dir: str,
        template: str,
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
    ) -> None:
        # Initialize the LLM client, output directory, template, and temperature
        self.client, self.model = create_client(model)
        self.output_dir = output_dir
        self.template = template
        self.temperature = temperature
        # Initialize the PaperSearchTool
        self.searcher: BaseTool = PaperSearchTool()
        # DrawerTool removed
        # self.drawer: BaseTool = DrawerTool(model, prompt_template_dir, temperature)
        self.formatter: BaseOutputFormatter
        self.config = Config(prompt_template_dir)
        # Initialize the output formatter based on the template (ACL or ICLR)
        if self.template == "acl":
            self.formatter = ACLOutputFormatter(model=self.model, client=self.client)
        elif self.template == "iclr":
            self.formatter = ICLROutputFormatter(model=self.model, client=self.client)

        # Load writer-specific prompts from the configuration
        self.prompts = self.config.prompt_template.writer_prompt

    def run(self, idea: Dict[str, Any]) -> Tuple[str, str]: # experiment_dir removed
        # This method orchestrates the paper writing process.
        # It no longer reads experiment logs or results.

        # Initialize dictionaries to store generated sections and references
        self.generated_sections: Dict[str, Any] = {}
        self.references: Dict[str, Any] = {}

        # Write the abstract based on the provided idea
        self._write_abstract(idea)

        # Define the sections to be written for this mini writer
        # Experiment-related sections are removed
        sections_to_write = [
            "Introduction",
            "Discussion",
            "Conclusion",
        ]

        for section in sections_to_write:
            # Write each section based on the idea
            self._write_section(idea, section)

        # Write the related work section using the PaperSearchTool
        self._write_related_work(idea)
        # Refine the entire paper for coherence and quality
        self._refine_paper()
        # Add citations to the relevant sections
        self._add_citations(idea)

        # Determine the paper name from the idea's title or use a default
        paper_name = idea.get("Title", "Research_Paper_Mini") # Changed default name
        # Sanitize paper name to be filesystem-friendly
        paper_name = re.sub(r'[\s\W]+', '_', paper_name)


        # Define the output PDF path
        output_pdf_path = f"{self.output_dir}/{paper_name}.pdf"
        # Use the formatter to generate the final LaTeX and PDF output
        self.formatter.run(
            content=self.generated_sections,
            references=self.references,
            output_dir=self.output_dir,
            output_pdf_path=output_pdf_path,
            name=paper_name,
        )
        # Return the path to the generated PDF and the paper name
        return output_pdf_path, paper_name

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        # This method generates the abstract for the paper.
        title = idea.get("Title", "Research Paper")

        # Format the prompt for generating the abstract
        abstract_prompt = self.prompts.abstract_prompt.format(
            abstract_tips=self.prompts.section_tips["Abstract"],
            title=title,
            problem=idea["Problem"],
            importance=idea["Importance"],
            difficulty=idea["Difficulty"],
            novelty=idea["NoveltyComparison"],
            experiment=idea["Experiment"], # 'Experiment' here refers to the proposed conceptual experiment from the idea
        )

        # Get the abstract content from the LLM
        abstract_content, _ = get_response_from_llm(
            msg=abstract_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt,
        )

        # Store the generated abstract
        self.generated_sections["Abstract"] = abstract_content

    # _generate_diagram_for_section method removed as DrawerTool is not used

    def _write_section(
        self,
        idea: Dict[str, Any],
        section: str,
    ) -> None:
        # This method generates content for a specific section of the paper.
        # It no longer takes experiment_log, experiment_result, or baseline_result.
        title = idea.get("Title", "Research Paper")
        # 'Experiment' from the idea is the conceptual experiment description
        experiment_description = idea.get("Experiment", "A novel conceptual approach.")
        print(f"Writing section: {section}...")

        # Prepare prompt based on the section type
        # Prompts are simplified as experimental data is not available
        if section == "Introduction":
            abstract_content = self.formatter.strip_latex(
                self.generated_sections.get("Abstract", "Abstract not yet generated.")
            )
            # The prompt for Introduction should focus on the problem, importance, and novelty from the idea.
            # It might refer to the conceptual experiment.
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                title=title,
                problem=idea["Problem"],
                importance=idea["Importance"],
                difficulty=idea["Difficulty"],
                novelty=idea["NoveltyComparison"],
                experiment=experiment_description,
                # method_section and experiment_results_summary removed
                abstract=abstract_content,
            )
        elif section == "Conclusion":
            # The Conclusion should summarize the core idea and its significance.
            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                experiment=experiment_description,
                # experiment_results_summary and method_summary removed
            )
        elif section == "Discussion":
            # The Discussion should explore implications, limitations, and future work based on the idea and related work.
            # It no longer uses experiment_results, baseline_results, or experiment_log.
            # It will rely on the conceptual 'experiment' from the idea and findings from 'Related_Work'.
            related_work_summary = self.formatter.strip_latex(
                self.generated_sections.get("Related_Work", "Related work section not yet generated.")
            )[:1500] # Provide a summary of related work

            section_prompt = self.prompts.section_prompt[section].format(
                section_tips=self.prompts.section_tips[section],
                experiment=experiment_description, # Conceptual experiment
                # baseline_results, experiment_results, experiment_log removed
                # Adding related_work_summary might be useful if the prompt supports it
                # For now, assuming the prompt for Discussion is general enough or will be adapted.
                # If the prompt strictly requires keys like 'experiment_results', this might error.
                # A more robust solution would be to have different prompt templates for writer_mini.
                # For now, we attempt with available data.
                # Prompt might be: experiment, problem, novelty, related_work_summary
                 problem=idea["Problem"],
                 novelty=idea["NoveltyComparison"],
                 related_work_summary=related_work_summary
            )
        else:
             # Default case for other sections (if any added later)
             # This part needs to be adapted if new non-experimental sections are introduced.
             try:
                 section_prompt = self.prompts.section_prompt[section].format(
                     section_tips=self.prompts.section_tips[section],
                     experiment=experiment_description,
                     title=title,
                     problem=idea["Problem"],
                     novelty=idea["NoveltyComparison"],
                     # Any other keys expected by a generic section prompt that are available from 'idea'
                 )
             except KeyError as e:
                 print(f"[ERROR] Prompt key missing for section '{section}': {e}. Prompt may need mini-specific version. Skipping.")
                 self.generated_sections[section] = f"Content for {section} could not be generated due to prompt configuration."
                 return


        # Get the section content from the LLM
        try:
            section_content, _ = get_response_from_llm(
                msg=section_prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.write_system_prompt,
            )
        except Exception as e:
            print(f"[ERROR] Failed to generate content for section {section}: {e}")
            section_content = f"An error occurred while generating content for {section}."

        # Diagram generation is removed.
        self.generated_sections[section] = section_content

    def _get_citations_related_work(
        self, idea: Dict[str, Any], num_cite_rounds: int, total_num_papers: int
    ) -> List[str]:
        # This method identifies relevant papers for the related work section using LLM.
        idea_title = idea.get("Title", "Research Paper")

        num_papers_per_round = (
            total_num_papers // num_cite_rounds
            if num_cite_rounds > 0
            else total_num_papers
        )
        collected_papers: List[str] = []

        # Iterate for a number of rounds to collect paper titles
        for round_num in range(num_cite_rounds):
            prompt = self.prompts.citation_related_work_prompt.format(
                idea_title=idea_title,
                problem=idea["Problem"],
                num_papers=num_papers_per_round,
                round_num=round_num + 1,
                collected_papers=collected_papers, # Provide already collected papers to avoid duplicates
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
                # Try parsing as direct JSON list
                potential_json = response.strip()
                if potential_json.startswith("[") and potential_json.endswith("]"):
                    new_titles = json.loads(potential_json)
                else:
                    # Fallback to extracting from markers like ```json ... ```
                    new_titles = extract_json_between_markers(response)
                    if isinstance(new_titles, dict) and len(new_titles) == 1:
                        key = list(new_titles.keys())[0]
                        if isinstance(new_titles[key], list):
                           new_titles = new_titles[key]
            except json.JSONDecodeError as e:
                 print(f"[WARNING] JSONDecodeError in citation round {round_num+1}: {e}. Response: {response[:200]}...")
                 # Fallback to simple regex for lines starting with common list markers
                 lines = response.split('\\n')
                 found_titles = [re.sub(r"^[ \-\d\.\*]+ ", "", line).strip() for line in lines if re.match(r"^[ \-\d\.\*]+ *\w", line)]
                 if found_titles:
                     print(f"[INFO] Extracted titles using regex fallback: {found_titles}")
                     new_titles = found_titles
                 else:
                    new_titles = None

            if new_titles and isinstance(new_titles, list):
                # Clean titles (remove potential numbering/bullets and quotes)
                cleaned_titles = [re.sub(r"^[ \-\d\.\*]+ ", "", str(t)).strip().strip('"') for t in new_titles if str(t).strip()]
                collected_papers.extend(cleaned_titles)
                collected_papers = list(dict.fromkeys(collected_papers)) # Remove duplicates
                print(f"Round {round_num+1}: Collected {len(cleaned_titles)} new unique titles.")
            else:
                print(f"Round {round_num+1}: No valid titles returned or extracted.")

            if len(collected_papers) >= total_num_papers:
                break # Stop if enough papers are collected
            time.sleep(1) # Pause to respect API rate limits

        print(f"Total unique related work titles collected: {len(collected_papers)}")
        return collected_papers[:total_num_papers] # Return up to the requested number

    def _search_reference(self, paper_list: List[str]) -> Dict[str, Any]:
        # This method uses the PaperSearchTool to find bibliographic information for a list of paper titles.
        results_dict = {}
        if not paper_list:
            return results_dict

        print(f"Searching references for {len(paper_list)} titles using PaperSearchTool...")
        processed_count = 0
        for paper_name in paper_list:
            if not paper_name or not isinstance(paper_name, str): # Skip invalid entries
                print(f"[INFO] Skipping invalid paper name: {paper_name}")
                continue
            try:
                # Use the PaperSearchTool
                result = self.searcher.run(paper_name) # searcher is PaperSearchTool

                if result:
                    # The result from PaperSearchTool is expected to be a dict where keys are matched paper titles
                    # and values are their metadata (including bibtex).
                    # We try to find the best match or take the first one.
                    found_match = False
                    normalized_search_name = paper_name.lower()
                    for key_from_tool_result in result:
                        if key_from_tool_result.lower() == normalized_search_name:
                             results_dict[key_from_tool_result] = result[key_from_tool_result]
                             found_match = True
                             break
                    if not found_match and result: # If no exact match, take the first result as a fallback
                        first_key = next(iter(result))
                        results_dict[first_key] = result[first_key]
                        print(f"[INFO] No exact title match for '{paper_name}', using first result from search: '{first_key}'")
                else:
                    print(f"[INFO] No search results from PaperSearchTool for '{paper_name}'.")


                processed_count += 1
                if processed_count % 5 == 0:
                     print(f"Searched {processed_count}/{len(paper_list)} references...")

                time.sleep(1.0) # Respect API rate limits
            except Exception as e:
                print(f"[ERROR] Error during reference search for '{paper_name}': {e}")
                # traceback.print_exc() # Enable for detailed debugging

        print(f"Reference search complete. Found bibliographic details for {len(results_dict)} titles.")
        return results_dict

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        # This method generates the "Related Work" section.
        print("Writing section: Related Work...")
        # First, get a list of potentially relevant paper titles using LLM
        citations_to_find = self._get_citations_related_work(
            idea, num_cite_rounds=2, total_num_papers=10 # Parameters for LLM-based title generation
        )

        # Then, search for these papers using PaperSearchTool to get bibtex and details
        paper_source_details = self._search_reference(citations_to_find)
        self.references = paper_source_details # Store all found reference details (including bibtex)

        if not paper_source_details:
            print("[WARNING] No references found by PaperSearchTool for Related Work section.")
            self.generated_sections["Related_Work"] = "No related work could be automatically retrieved for this topic."
            return

        # Prepare a list of found references for the LLM prompt that will write the section's narrative
        reference_prompt_list = []
        # This map will help link LLM's textual citations back to actual bibtex keys
        self.title_to_bibtex_key_map: Dict[str,str] = {}

        for title, meta in paper_source_details.items():
            bibtex_entry = meta.get("bibtex", "")
            # Extract bibtex key
            bibtex_key_match = re.search(r"@\w+\{(.+?),", bibtex_entry)
            bibtex_key = bibtex_key_match.group(1) if bibtex_key_match else title.lower().replace(" ", "_")[:20] # Fallback key
            # Extract year
            year_match = re.search(r"year\s*=\s*\{?(\d{4})\}?", bibtex_entry, re.IGNORECASE)
            year = year_match.group(1) if year_match else "N/A"

            reference_prompt_list.append(f"- {title} (Ref Key: {bibtex_key}, Year: {year})")
            self.title_to_bibtex_key_map[title.lower()] = bibtex_key # Store mapping for later citation
            self.title_to_bibtex_key_map[bibtex_key.lower()] = bibtex_key # Also map key to key for robustness

        reference_list_str_for_prompt = "\n".join(reference_prompt_list)
        experiment_description = idea.get("Experiment", "A novel conceptual approach.") # Conceptual experiment

        # Format the prompt for writing the related work narrative
        related_work_prompt = self.prompts.related_work_prompt.format(
            related_work_tips=self.prompts.section_tips["Related_Work"],
            experiment=experiment_description,
            references=reference_list_str_for_prompt, # Provide the list of found papers with their bibtex keys
        )

        # Get the related work content from the LLM
        relatedwork_content, _ = get_response_from_llm(
            msg=related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt_related_work,
        )

        # Post-process the generated content to replace placeholders like \\cite{Title...} or \\cite{Ref Key: ...}
        # with actual \\cite{bibtex_key} using the title_to_bibtex_key_map.
        def replace_citation_placeholder(match_obj):
            placeholder_text = match_obj.group(1).strip()
            # Try matching by "Ref Key: actual_key"
            ref_key_match = re.match(r"Ref Key:\s*(\S+)", placeholder_text, re.IGNORECASE)
            if ref_key_match:
                potential_key = ref_key_match.group(1).lower()
                if potential_key in self.title_to_bibtex_key_map:
                    return f"\\\\cite{{{self.title_to_bibtex_key_map[potential_key]}}}"

            # Try matching by title (case-insensitive)
            normalized_placeholder = placeholder_text.lower()
            if normalized_placeholder in self.title_to_bibtex_key_map:
                return f"\\\\cite{{{self.title_to_bibtex_key_map[normalized_placeholder]}}}"
            
            # Fallback: if the placeholder itself is a direct bibtex key
            if placeholder_text in self.title_to_bibtex_key_map.values():
                 return f"\\\\cite{{{placeholder_text}}}"


            # A more fuzzy match if the LLM used part of the title
            for title_from_map, key_from_map in self.title_to_bibtex_key_map.items():
                if placeholder_text.lower() in title_from_map or title_from_map in placeholder_text.lower() : # simple substring
                    print(f"[INFO] Heuristic citation match for '{placeholder_text}' -> '{key_from_map}'")
                    return f"\\\\cite{{{key_from_map}}}"
            
            print(f"[WARNING] Could not map citation placeholder '{placeholder_text}' in Related Work to a bibtex key.")
            return f"\\\\cite{{{placeholder_text}}} % FIXME: Unmapped citation"

        # This regex looks for \cite{ANYTHING_INSIDE_BRACES}
        relatedwork_content_with_citations = re.sub(r"\\cite\{(.+?)\}", replace_citation_placeholder, relatedwork_content)

        self.generated_sections["Related_Work"] = relatedwork_content_with_citations


    def _refine_paper(self) -> None:
        # This method refines all generated sections of the paper.
        print("Refining paper sections (mini version)...")
        # Combine sections into a draft for context.
        # Exclude Title from this draft if it's handled separately or added by formatter.
        draft_sections_for_context = []
        for section_name, section_content in self.generated_sections.items():
             if section_name != "Title": # Title is often refined separately or handled by formatter
                 # For context, it's better to have plain text
                 plain_text_content = self.formatter.strip_latex(section_content)
                 draft_sections_for_context.append(f"\\section{{{section_name}}}\\n\\n{plain_text_content}")

        full_draft_context = "\n\n".join(draft_sections_for_context)
        # Truncate if too long to avoid issues with LLM context limits
        if len(full_draft_context) > 15000:
             print(f"[INFO] Full draft for refinement context is long ({len(full_draft_context)} chars), truncating.")
             full_draft_context = full_draft_context[:7500] + "\n... [DRAFT TRUNCATED FOR CONTEXT] ...\n" + full_draft_context[-7500:]

        # Attempt to refine the title using the draft context
        try:
            print("Refining title (mini version)...")
            if "title_refinement_prompt" in self.prompts:
                title_refinement_prompt = self.prompts.title_refinement_prompt.format(full_draft=full_draft_context)
                refined_title, _ = get_response_from_llm(
                    msg=title_refinement_prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts.write_system_prompt, # General writing system prompt
                )
                # Clean the refined title
                refined_title = refined_title.strip().strip('"').strip()
                refined_title = re.sub(r"\*\*(.*?)\*\*", r"\1", refined_title) # Remove markdown bold
                if refined_title:
                     self.generated_sections["Title"] = refined_title # Store the refined title
                     print(f"Refined Title: {refined_title}")
            else:
                print("[INFO] title_refinement_prompt not found, skipping title refinement.")
        except KeyError as e:
            print(f"[WARNING] Missing key '{e}' in title_refinement_prompt. Skipping title refinement.")
        except Exception as e:
            print(f"[ERROR] Failed to refine title: {e}")
            traceback.print_exc()


        # Define sections to refine for the mini writer
        sections_to_refine = [
            "Abstract",
            "Introduction",
            "Related_Work", # Related work also benefits from refinement
            "Discussion",
            "Conclusion",
        ]

        for section_name in sections_to_refine:
            if section_name in self.generated_sections:
                print(f"Refining section: {section_name} (mini version)...")
                original_section_content = self.generated_sections[section_name]

                # Use a general refinement prompt or a section-specific one if available
                # Assuming 'second_refinement_prompt' is the primary one for sections
                if "second_refinement_prompt" in self.prompts:
                    try:
                        # Get section-specific tips or general tips
                        section_tips = self.prompts.section_tips.get(section_name, "General academic writing tips.")
                        # Get a list of common errors to avoid
                        error_list_for_prompt = self.prompts.get("error_list", "- Vague statements\n- Grammatical errors")

                        # Format the refinement prompt
                        current_refinement_prompt = self.prompts.second_refinement_prompt.format(
                            section=section_name,
                            tips=section_tips,
                            full_draft=full_draft_context, # Provide the (potentially truncated) draft as context
                            section_content=original_section_content, # The original content of the section to refine
                            error_list=error_list_for_prompt,
                        )

                        refined_section_content, _ = get_response_from_llm(
                            msg=current_refinement_prompt,
                            client=self.client,
                            model=self.model,
                            system_message=self.prompts.write_system_prompt, # General writing system prompt
                            temperature=0.5 # Lower temperature for more focused refinement
                        )

                        # Basic check to ensure refinement didn't fail or return empty/too short content
                        if refined_section_content and len(refined_section_content) > 0.5 * len(original_section_content):
                            self.generated_sections[section_name] = refined_section_content
                        else:
                             print(f"[WARNING] Refinement for {section_name} produced suspiciously short or empty result. Keeping original.")
                    except KeyError as e:
                         print(f"[WARNING] Missing key '{e}' in prompt format for refining {section_name}. Skipping this refinement.")
                    except Exception as e:
                         print(f"[ERROR] Failed to refine section {section_name}: {e}")
                         traceback.print_exc()
                else:
                    print(f"[INFO] 'second_refinement_prompt' not found. Skipping refinement for {section_name}.")
            else:
                print(f"[INFO] Section {section_name} not found for refinement. Skipping.")


    def _add_citations(self, idea: Dict[str, Any]) -> None:
        # This method attempts to add \\cite{key} markers into the generated sections
        # using the references already found and stored in self.references.
        print("Adding citations to sections (mini version)...")
        
        # Use existing references gathered during related work generation (self.references should have bibtex keys)
        # self.title_to_bibtex_key_map should also be populated from _write_related_work
        if not self.references or not hasattr(self, 'title_to_bibtex_key_map') or not self.title_to_bibtex_key_map:
             print("[INFO] No existing references or bibtex key map found to add citations from. Skipping citation embedding.")
             return

        # Prepare a list of available references (Title, Ref Key, Year) for the LLM prompt
        reference_prompt_list_for_embedding = []
        for title, meta_info in self.references.items():
            bibtex_entry = meta_info.get("bibtex", "")
            bibtex_key_match = re.search(r"@\w+\{(.+?),", bibtex_entry)
            if bibtex_key_match:
                bibtex_key = bibtex_key_match.group(1)
                year_match = re.search(r"year\s*=\s*\{?(\d{4})\}?", bibtex_entry, re.IGNORECASE)
                year_str = year_match.group(1) if year_match else "N/A"
                reference_prompt_list_for_embedding.append(f"- {title} (Ref Key: {bibtex_key}, Year: {year_str})")
            # else: title already logged as problematic in _write_related_work if key extraction failed

        reference_list_str_for_embedding_prompt = "\n".join(reference_prompt_list_for_embedding)
        if not reference_list_str_for_embedding_prompt:
            print("[INFO] No references formatted for embedding prompt. Skipping citation embedding.")
            return

        # Define sections where citations are most relevant for a conceptual paper
        sections_to_add_citations_to = ["Introduction", "Related_Work", "Discussion"]

        for section_name in sections_to_add_citations_to:
            if section_name in self.generated_sections:
                print(f"Attempting to add/embed citations in: {section_name}...")
                try:
                    original_section_content = self.generated_sections[section_name]

                    if "embed_citation_prompt" not in self.prompts:
                        print(f"[WARNING] 'embed_citation_prompt' not found in prompts. Cannot add citations to {section_name}.")
                        continue

                    # Format the prompt to ask LLM to embed citations
                    citation_embedding_prompt = self.prompts.embed_citation_prompt.format(
                        section=section_name,
                        section_content=original_section_content, # Provide the current content of the section
                        references=reference_list_str_for_embedding_prompt, # Provide the list of available references
                    )

                    # Get the section content with embedded citation placeholders from LLM
                    section_with_citation_placeholders, _ = get_response_from_llm(
                        msg=citation_embedding_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts.citation_system_prompt, # A system prompt emphasizing careful citation
                        temperature=0.2 # Low temperature for precision in citation
                    )

                    # Post-processing: Replace LLM's citation placeholders (e.g., \\cite{Title} or \\cite{Ref Key: X})
                    # with actual \\cite{actual_bibtex_key} using self.title_to_bibtex_key_map
                    def replace_embedded_citation_placeholder(match_obj):
                        placeholder_text = match_obj.group(1).strip()
                        # Try matching by "Ref Key: actual_key"
                        ref_key_match = re.search(r"Ref Key:\s*(\S+)", placeholder_text, re.IGNORECASE)
                        if ref_key_match:
                            potential_key = ref_key_match.group(1).lower()
                            if potential_key in self.title_to_bibtex_key_map: # Map might hold key.lower() -> actual_key
                                return f"\\\\cite{{{self.title_to_bibtex_key_map[potential_key]}}}"
                        
                        # Try matching by title (case-insensitive)
                        normalized_placeholder = placeholder_text.lower()
                        if normalized_placeholder in self.title_to_bibtex_key_map:
                            return f"\\\\cite{{{self.title_to_bibtex_key_map[normalized_placeholder]}}}"
                        
                        # Fallback: if the placeholder itself is a direct bibtex key (already correctly formatted by LLM)
                        if placeholder_text in self.title_to_bibtex_key_map.values():
                            return f"\\\\cite{{{placeholder_text}}}"

                        # Last resort: fuzzy match against titles in the map
                        for title_from_map, key_from_map in self.title_to_bibtex_key_map.items():
                             if placeholder_text.lower() in title_from_map or title_from_map in placeholder_text.lower():
                                 print(f"[INFO] Heuristic citation match (embed) for '{placeholder_text}' -> '{key_from_map}'")
                                 return f"\\\\cite{{{key_from_map}}}"

                        print(f"[WARNING] Could not map embedded citation placeholder '{placeholder_text}' in {section_name}.")
                        return f"\\\\cite{{{placeholder_text}}} % FIXME: Unmapped embedded citation"

                    # Apply the replacement to the section content
                    final_section_with_citations = re.sub(r"\\cite\{(.+?)\}", replace_embedded_citation_placeholder, section_with_citation_placeholders)

                    # Basic check: ensure content wasn't drastically changed or deleted by citation process
                    if len(final_section_with_citations) > 0.7 * len(original_section_content):
                         self.generated_sections[section_name] = final_section_with_citations
                         original_cites = len(re.findall(r"\\cite\{.*?\}", original_section_content))
                         new_cites = len(re.findall(r"\\cite\{.*?\}", final_section_with_citations))
                         print(f"[INFO] Citations in {section_name} after embedding: {original_cites} -> {new_cites}")
                    else:
                         print(f"[WARNING] Citation embedding for {section_name} significantly changed content length or content. Reverting to original.")

                except KeyError as e:
                    print(f"[WARNING] Missing key '{e}' in prompt format for adding citations to {section_name}.")
                except Exception as e:
                    print(f"[ERROR] Failed to add citations to section {section_name}: {e}")
                    traceback.print_exc()
            else:
                print(f"[INFO] Section {section_name} not found for citation embedding. Skipping.") 