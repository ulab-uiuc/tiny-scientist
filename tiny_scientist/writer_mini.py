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
# OutputFormatter related imports are removed
# from .utils.output_formatter import (
#     ACLOutputFormatter,
#     BaseOutputFormatter,
#     ICLROutputFormatter,
# )

# Removed cairosvg import as DrawerTool is removed


class WriterMini: # Renamed class
    def __init__(
        self,
        model: str,
        output_dir: str, # output_dir might be used for saving intermediate files or logs if any
        template: str, # template might still influence prompt selection or other logic
        temperature: float = 0.75,
        prompt_template_dir: Optional[str] = None,
    ) -> None:
        # Initialize the LLM client, output directory, template, and temperature
        self.client, self.model = create_client(model)
        self.output_dir = output_dir # Retained for potential future use (e.g. saving text file)
        self.template = template # Retained for potential future use
        self.temperature = temperature
        # Initialize the PaperSearchTool
        self.searcher: BaseTool = PaperSearchTool()
        # DrawerTool removed
        # self.formatter is removed
        self.config = Config(prompt_template_dir)
        # Output formatter initialization is removed

        # Load writer-specific prompts from the configuration
        self.prompts = self.config.prompt_template.writer_prompt

    def run(self, idea: Dict[str, Any]) -> str: # Return type changed to str
        # This method orchestrates the paper writing process.
        # It no longer produces a PDF or LaTeX output.

        # Initialize dictionaries to store generated sections and references
        self.generated_sections: Dict[str, Any] = {}
        self.references: Dict[str, Any] = {}

        # Add a high-level debug log for the received idea
        print(f"[DEBUG] WriterMini.run: Received idea with keys: {list(idea.keys())}")
        print(f"[DEBUG] WriterMini.run: Idea content (first 500 chars): {json.dumps(idea, indent=2)[:500]}...")

        # Check for absolutely critical keys at the start of run, though Thinker should handle this
        critical_keys_for_writer = ["Title", "Problem"] # Example minimal set for writer to even start
        for key in critical_keys_for_writer:
            if key not in idea:
                print(f"[ERROR] WriterMini.run: Critical key '{key}' missing in idea. This should have been handled by Thinker. Using placeholder.")
                idea[key] = f"Placeholder for missing critical key: {key}"

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

        # Compile all sections into a single text string
        final_text_output = self._compile_text_output()
        
        # Optionally, save the text output to a file in output_dir
        # paper_name_for_file = idea.get("Title", "Research_Paper_Mini_Text")
        # paper_name_for_file = re.sub(r'[\\s\\W]+', '_', paper_name_for_file)
        # text_file_path = os.path.join(self.output_dir, f"{paper_name_for_file}.txt")
        # try:
        #     with open(text_file_path, 'w', encoding='utf-8') as f:
        #         f.write(final_text_output)
        #     print(f"[INFO] Text output saved to {text_file_path}")
        # except Exception as e:
        #     print(f"[WARNING] Could not save text output to file: {e}")


        return final_text_output # Return the combined text string

    def _compile_text_output(self) -> str:
        """Compiles all generated sections into a single string with markdown headers."""
        ordered_sections = [
            ("Title", "# {content}\\n"), # Main title
            ("Abstract", "## Abstract\\n\\n{content}\\n"),
            ("Introduction", "## Introduction\\n\\n{content}\\n"),
            ("Related_Work", "## Related Work\\n\\n{content}\\n"),
            ("Discussion", "## Discussion\\n\\n{content}\\n"),
            ("Conclusion", "## Conclusion\\n\\n{content}\\n"),
        ]

        paper_content_parts = []
        for section_key, section_format_str in ordered_sections:
            content = self.generated_sections.get(section_key, "")
            if content: # Add section if content exists
                 # A basic attempt to remove some common LaTeX sectioning commands if they are part of content
                 content = re.sub(r"\\section\*?\{.+?\}", "", content).strip()
                 content = re.sub(r"\\subsection\*?\{.+?\}", "", content).strip()
                 paper_content_parts.append(section_format_str.format(content=content))
        
        full_paper_text = "\\n".join(paper_content_parts)

        # Append References
        if self.references:
            full_paper_text += "\\n## References\\n\\n"
            ref_list = []
            for i, (title, meta) in enumerate(self.references.items()):
                bibtex_entry = meta.get("bibtex", "")
                bibtex_key_match = re.search(r"@\\w+\\{(.+?),", bibtex_entry)
                bibtex_key = bibtex_key_match.group(1) if bibtex_key_match else f"ref_{i+1}"
                
                # For simplicity, just list Title (BibTeX Key) or a snippet of BibTeX.
                # A full plain text conversion of BibTeX is complex.
                # ref_list.append(f"- {title} (Key: {bibtex_key})") 
                # Or append the bibtex entry itself
                ref_list.append(f"### {title} (Key: {bibtex_key})\\n```bibtex\\n{bibtex_entry}\\n```\\n")
            full_paper_text += "\\n".join(ref_list)
            
        return full_paper_text.strip()

    def _write_abstract(self, idea: Dict[str, Any]) -> None:
        # This method generates the abstract for the paper.
        print("[INFO] WriterMini._write_abstract: Generating abstract.")
        # Safely get all required values from the idea dict, providing defaults
        title = idea.get("Title", "Default Title: Research Paper")
        problem = idea.get("Problem", "Problem not specified in idea.")
        importance = idea.get("Importance", "Importance not specified.")
        difficulty = idea.get("Difficulty", "Difficulty not specified.")
        novelty_comparison = idea.get("NoveltyComparison", "Novelty comparison not specified.")
        
        # Handle the 'Experiment' field, ensuring it's a dict and has a 'Description'
        experiment_data = idea.get("Experiment")
        if isinstance(experiment_data, dict):
            experiment_description = experiment_data.get("Description", "Conceptual experiment description not specified.")
        elif isinstance(experiment_data, str): # If it's a string, use it directly but log a warning
            experiment_description = experiment_data
            print(f"[WARNING] WriterMini._write_abstract: 'Experiment' field in idea is a string, expected a dict. Using string content: {experiment_data[:100]}...")
        else:
            experiment_description = "Conceptual experiment not specified or invalid format."
            print(f"[WARNING] WriterMini._write_abstract: 'Experiment' field in idea is missing or has an unexpected type. Value: {experiment_data}")

        # Assume 'abstract_tips' comes from self.prompts.section_tips, not directly from 'idea' for this prompt
        abstract_tips_content = self.prompts.section_tips.get("Abstract", "Provide a concise summary of the research.")

        try:
            abstract_prompt = self.prompts.abstract_prompt.format(
                abstract_tips=abstract_tips_content,
                title=title,
                problem=problem,
                importance=importance,
                difficulty=difficulty,
                novelty=novelty_comparison,
                experiment=experiment_description
            )
        except KeyError as e:
            print(f"[ERROR] WriterMini._write_abstract: KeyError formatting abstract_prompt: {e}. This indicates a mismatch between prompt template and provided keys.")
            print(f"[DEBUG] Available prompt keys in abstract_prompt template vs. provided arguments.")
            # Fallback to a very basic prompt if formatting fails catastrophically
            abstract_prompt = f"Write an abstract for a paper titled '{title}' about '{problem}'."
            self.generated_sections["Abstract"] = "Error generating abstract due to prompt formatting issues."
            self.generated_sections["Title"] = title
            return

        abstract_content, _ = get_response_from_llm(
            msg=abstract_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt,
        )
        self.generated_sections["Abstract"] = abstract_content or "Abstract could not be generated."
        self.generated_sections["Title"] = title # Ensure title is stored
        print("[INFO] WriterMini._write_abstract: Abstract generation complete.")

    def _write_section(
        self,
        idea: Dict[str, Any],
        section: str,
    ) -> None:
        # This method generates content for a specific section of the paper.
        title = idea.get("Title", "Research Paper")
        # Safely get experiment description
        experiment_data = idea.get("Experiment")
        if isinstance(experiment_data, dict):
            experiment_description = experiment_data.get("Description", "A novel conceptual approach.")
        else:
            experiment_description = "Experiment details not available in the correct format."
        
        print(f"Writing section: {section}...")
        section_prompt_template_str = self.prompts.section_prompt.get(section)
        if not section_prompt_template_str:
            print(f"[ERROR] Prompt template for section '{section}' not found. Skipping.")
            self.generated_sections[section] = f"Content for {section} could not be generated: Prompt template missing."
            return

        format_args = {
            "section_tips": self.prompts.section_tips.get(section, "General writing tips."),
            "title": title,
            "problem": idea.get("Problem", "Problem not specified."),
            "importance": idea.get("Importance", "Importance not specified."),
            "difficulty": idea.get("Difficulty", "Difficulty not specified."),
            "novelty": idea.get("NoveltyComparison", "Novelty comparison not specified."),
            "experiment": experiment_description,
            # Ensure other keys potentially used by section prompts are also safely accessed
            "abstract": self.generated_sections.get("Abstract", "Abstract not yet generated."),
            "related_work_summary": self.generated_sections.get("Related_Work", "Related work not yet generated.")[:1500],
        }

        if section == "Introduction":
            format_args["abstract"] = self.generated_sections.get("Abstract", "Abstract not yet generated.")
            # Removed method_section as per previous fixes and WriterMini scope
        elif section == "Conclusion":
            pass # Uses common args
        elif section == "Discussion":
            # Use raw related work summary as context (no formatter.strip_latex)
            related_work_summary = self.generated_sections.get("Related_Work", "Related work section not yet generated.")[:1500]
            format_args["related_work_summary"] = related_work_summary
            # Ensure prompt for Discussion in YAML is adapted for conceptual papers and uses these keys
            # E.g., experiment, problem, novelty, related_work_summary
            # Removing experiment_results, baseline_results from args as they are not available
        
        # Filter args to only those present in the template to avoid KeyErrors
        # This is a safer approach than providing all possible keys
        # actual_args_for_format = {} # Old logic removed
        # # Find all placeholder keys in the prompt string
        # expected_keys = re.findall(r'\{([^}]+)\}', section_prompt_template_str)
        # 
        # for key in expected_keys:
        #     if key in format_args:
        #         actual_args_for_format[key] = format_args[key]
        #     else:
        #         # Provide a default or skip if a key in template is not in format_args
        #         # This indicates a mismatch that should ideally be fixed in prompts or arg preparation
        #         print(f"[WARNING] Prompt for section '{section}' expects key '{key}' which was not provided. Using empty string.")
        #         actual_args_for_format[key] = "" 
        
        try:
            # Pass the prepared format_args directly.
            # If the prompt template string (section_prompt_template_str) contains a placeholder
            # (e.g., "{some_key}") that is NOT a key in format_args, .format() will raise a KeyError.
            # This is the desired behavior as it correctly identifies missing data or incorrect prompt templates.
            section_prompt = section_prompt_template_str.format(**format_args)
        except KeyError as e:
            # print(f"[ERROR] Still a KeyError formatting prompt for section '{section}': {e}. Args: {actual_args_for_format.keys()}")
            print(f"[ERROR] WriterMini._write_section: KeyError formatting prompt for section '{section}': {e}. Args: {list(format_args.keys())}")
            self.generated_sections[section] = f"Content for {section} could not be generated due to prompt formatting error: Missing key {e}."
            return

        try:
            section_content, _ = get_response_from_llm(
                msg=section_prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.write_system_prompt, # This system prompt might need adjustment for text output
            )
        except Exception as e:
            print(f"[ERROR] WriterMini._write_section: Failed to generate content for section {section}: {e}")
            section_content = f"An error occurred while generating content for {section}."

        self.generated_sections[section] = section_content

    def _get_citations_related_work(
        self, idea: Dict[str, Any], num_cite_rounds: int, total_num_papers: int
    ) -> List[str]:
        idea_title = idea.get("Title", "Research Paper")
        problem_desc = idea.get("Problem", "No specific problem described.") # Safe get
        num_papers_per_round = (
            total_num_papers // num_cite_rounds
            if num_cite_rounds > 0
            else total_num_papers
        )
        collected_papers: List[str] = []

        for round_num in range(num_cite_rounds):
            prompt = self.prompts.citation_related_work_prompt.format(
                idea_title=idea_title,
                problem=problem_desc, # Use safe variable
                num_papers=num_papers_per_round,
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
                potential_json = response.strip()
                if potential_json.startswith("[") and potential_json.endswith("]"):
                    new_titles = json.loads(potential_json)
                else:
                    new_titles = extract_json_between_markers(response)
                    if isinstance(new_titles, dict) and len(new_titles) == 1:
                        key = list(new_titles.keys())[0]
                        if isinstance(new_titles[key], list):
                           new_titles = new_titles[key]
            except json.JSONDecodeError as e:
                 print(f"[WARNING] JSONDecodeError in citation round {round_num+1}: {e}. Response: {response[:200]}...")
                 lines = response.split('\\\\n')
                 found_titles = [re.sub(r"^[ \\-\\d\\.\\*]+ ", "", line).strip() for line in lines if re.match(r"^[ \\-\\d\\.\\*]+ *\\w", line)]
                 if found_titles:
                     print(f"[INFO] Extracted titles using regex fallback: {found_titles}")
                     new_titles = found_titles
                 else:
                    new_titles = None

            if new_titles and isinstance(new_titles, list):
                cleaned_titles = [re.sub(r"^[ \\-\\d\\.\\*]+ ", "", str(t)).strip().strip('"') for t in new_titles if str(t).strip()]
                collected_papers.extend(cleaned_titles)
                collected_papers = list(dict.fromkeys(collected_papers))
                print(f"Round {round_num+1}: Collected {len(cleaned_titles)} new unique titles.")
            else:
                print(f"Round {round_num+1}: No valid titles returned or extracted.")

            if len(collected_papers) >= total_num_papers:
                break 
            time.sleep(1) 

        print(f"Total unique related work titles collected: {len(collected_papers)}")
        return collected_papers[:total_num_papers]

    def _search_reference(self, paper_list: List[str]) -> Dict[str, Any]:
        results_dict = {}
        if not paper_list:
            return results_dict

        print(f"Searching references for {len(paper_list)} titles using PaperSearchTool...")
        processed_count = 0
        for paper_name in paper_list:
            if not paper_name or not isinstance(paper_name, str):
                print(f"[INFO] Skipping invalid paper name: {paper_name}")
                continue
            try:
                result = self.searcher.run(paper_name) 

                if result:
                    found_match = False
                    normalized_search_name = paper_name.lower()
                    for key_from_tool_result in result:
                        if key_from_tool_result.lower() == normalized_search_name:
                             results_dict[key_from_tool_result] = result[key_from_tool_result]
                             found_match = True
                             break
                    if not found_match and result: 
                        first_key = next(iter(result))
                        results_dict[first_key] = result[first_key]
                        print(f"[INFO] No exact title match for '{paper_name}', using first result from search: '{first_key}'")
                else:
                    print(f"[INFO] No search results from PaperSearchTool for '{paper_name}'.")

                processed_count += 1
                if processed_count % 5 == 0:
                     print(f"Searched {processed_count}/{len(paper_list)} references...")
                time.sleep(1.0) 
            except Exception as e:
                print(f"[ERROR] Error during reference search for '{paper_name}': {e}")
        print(f"Reference search complete. Found bibliographic details for {len(results_dict)} titles.")
        return results_dict

    def _write_related_work(self, idea: Dict[str, Any]) -> None:
        print("Writing section: Related Work...")
        citations_to_find = self._get_citations_related_work(
            idea, num_cite_rounds=2, total_num_papers=10
        )
        paper_source_details = self._search_reference(citations_to_find)
        self.references = paper_source_details

        if not paper_source_details:
            print("[WARNING] No references found by PaperSearchTool for Related Work section.")
            self.generated_sections["Related_Work"] = "No related work could be automatically retrieved for this topic."
            return

        reference_prompt_list = []
        self.title_to_bibtex_key_map: Dict[str,str] = {}

        for title, meta in paper_source_details.items():
            bibtex_entry = meta.get("bibtex", "")
            bibtex_key_match = re.search(r"@\\w+\\{(.+?),", bibtex_entry)
            bibtex_key = bibtex_key_match.group(1) if bibtex_key_match else title.lower().replace(" ", "_")[:20]
            year_match = re.search(r"year\\s*=\\s*\\{?(\\d{4})\\}?", bibtex_entry, re.IGNORECASE)
            year = year_match.group(1) if year_match else "N/A"

            reference_prompt_list.append(f"- {title} (Ref Key: {bibtex_key}, Year: {year})")
            self.title_to_bibtex_key_map[title.lower()] = bibtex_key
            self.title_to_bibtex_key_map[bibtex_key.lower()] = bibtex_key

        reference_list_str_for_prompt = "\\n".join(reference_prompt_list)
        experiment_description = idea.get("Experiment", "A novel conceptual approach.")

        related_work_prompt = self.prompts.related_work_prompt.format(
            related_work_tips=self.prompts.section_tips["Related_Work"],
            experiment=experiment_description,
            references=reference_list_str_for_prompt,
        )

        relatedwork_content, _ = get_response_from_llm(
            msg=related_work_prompt,
            client=self.client,
            model=self.model,
            system_message=self.prompts.write_system_prompt_related_work, # This system prompt might need adjustment for text output
        )
        
        def replace_citation_placeholder(match_obj):
            placeholder_text = match_obj.group(1).strip()
            ref_key_match = re.match(r"Ref Key:\\s*(\\S+)", placeholder_text, re.IGNORECASE)
            if ref_key_match:
                potential_key = ref_key_match.group(1).lower()
                if potential_key in self.title_to_bibtex_key_map:
                    return f"\\\\cite{{{self.title_to_bibtex_key_map[potential_key]}}}"

            normalized_placeholder = placeholder_text.lower()
            if normalized_placeholder in self.title_to_bibtex_key_map:
                return f"\\\\cite{{{self.title_to_bibtex_key_map[normalized_placeholder]}}}"
            
            if placeholder_text in self.title_to_bibtex_key_map.values():
                 return f"\\\\cite{{{placeholder_text}}}"

            for title_from_map, key_from_map in self.title_to_bibtex_key_map.items():
                if placeholder_text.lower() in title_from_map or title_from_map in placeholder_text.lower() :
                    print(f"[INFO] Heuristic citation match for '{placeholder_text}' -> '{key_from_map}'")
                    return f"\\\\cite{{{key_from_map}}}"
            
            print(f"[WARNING] Could not map citation placeholder '{placeholder_text}' in Related Work to a bibtex key.")
            return f"\\\\cite{{{placeholder_text}}} % FIXME: Unmapped citation"

        relatedwork_content_with_citations = re.sub(r"\\\\cite\\{(.+?)\\}", replace_citation_placeholder, relatedwork_content)
        self.generated_sections["Related_Work"] = relatedwork_content_with_citations


    def _refine_paper(self) -> None:
        print("Refining paper sections (text version)...")
        draft_sections_for_context = []
        # Define order for context compilation
        ordered_context_keys = ["Title", "Abstract", "Introduction", "Related_Work", "Discussion", "Conclusion"]
        for section_name in ordered_context_keys:
            if section_name in self.generated_sections:
                 # No formatter.strip_latex(), use raw (potentially text/markdown) content
                 section_content = self.generated_sections[section_name]
                 draft_sections_for_context.append(f"## {section_name}\\n\\n{section_content}")

        full_draft_context = "\\n\\n".join(draft_sections_for_context)
        if len(full_draft_context) > 15000: # Truncate if too long
             print(f"[INFO] Full draft for refinement context is long ({len(full_draft_context)} chars), truncating.")
             full_draft_context = full_draft_context[:7500] + "\\n... [DRAFT TRUNCATED FOR CONTEXT] ...\\n" + full_draft_context[-7500:]

        try:
            print("Refining title (text version)...")
            if "title_refinement_prompt" in self.prompts:
                # This prompt should ask for plain text title
                title_refinement_prompt = self.prompts.title_refinement_prompt.format(full_draft=full_draft_context)
                refined_title, _ = get_response_from_llm(
                    msg=title_refinement_prompt,
                    client=self.client,
                    model=self.model,
                    system_message=self.prompts.write_system_prompt, # Ensure this system prompt is suitable for text
                )
                refined_title = refined_title.strip().strip('"').strip()
                refined_title = re.sub(r"\\*\\*(.*?)\\*\\*", r"\\1", refined_title) 
                if refined_title:
                     self.generated_sections["Title"] = refined_title
                     print(f"Refined Title: {refined_title}")
            else:
                print("[INFO] title_refinement_prompt not found, skipping title refinement.")
        except KeyError as e:
            print(f"[WARNING] Missing key '{e}' in title_refinement_prompt. Skipping title refinement.")
        except Exception as e:
            print(f"[ERROR] Failed to refine title: {e}")
            traceback.print_exc()

        sections_to_refine = ["Abstract", "Introduction", "Related_Work", "Discussion", "Conclusion"]
        for section_name in sections_to_refine:
            if section_name in self.generated_sections:
                print(f"Refining section: {section_name} (text version)...")
                original_section_content = self.generated_sections[section_name]

                if "second_refinement_prompt" in self.prompts: # Assuming this prompt asks for text/markdown
                    try:
                        section_tips = self.prompts.section_tips.get(section_name, "General academic writing tips.")
                        error_list_for_prompt = self.prompts.get("error_list", "- Vague statements\\n- Grammatical errors")

                        current_refinement_prompt = self.prompts.second_refinement_prompt.format(
                            section=section_name,
                            tips=section_tips,
                            full_draft=full_draft_context,
                            section_content=original_section_content,
                            error_list=error_list_for_prompt,
                        )
                        # The refinement prompt should ideally instruct LLM to output clean text/markdown
                        refined_section_content, _ = get_response_from_llm(
                            msg=current_refinement_prompt,
                            client=self.client,
                            model=self.model,
                            system_message=self.prompts.write_system_prompt,
                            temperature=0.5 
                        )
                        if refined_section_content and len(refined_section_content) > 0.5 * len(original_section_content):
                            self.generated_sections[section_name] = refined_section_content
                        else:
                             print(f"[WARNING] Refinement for {section_name} produced suspiciously short/empty result. Keeping original.")
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
        # This method attempts to add \\cite{key} markers into the generated sections (text/markdown)
        print("Adding citations to text sections...")
        
        if not self.references or not hasattr(self, 'title_to_bibtex_key_map') or not self.title_to_bibtex_key_map:
             print("[INFO] No existing references or bibtex key map found to add citations from. Skipping citation embedding.")
             return

        reference_prompt_list_for_embedding = []
        for title, meta_info in self.references.items():
            bibtex_entry = meta_info.get("bibtex", "")
            bibtex_key_match = re.search(r"@\\w+\\{(.+?),", bibtex_entry)
            if bibtex_key_match:
                bibtex_key = bibtex_key_match.group(1)
                year_match = re.search(r"year\\s*=\\s*\\{?(\\d{4})\\}?", bibtex_entry, re.IGNORECASE)
                year_str = year_match.group(1) if year_match else "N/A"
                reference_prompt_list_for_embedding.append(f"- {title} (Ref Key: {bibtex_key}, Year: {year_str})")

        reference_list_str_for_embedding_prompt = "\\n".join(reference_prompt_list_for_embedding)
        if not reference_list_str_for_embedding_prompt:
            print("[INFO] No references formatted for embedding prompt. Skipping citation embedding.")
            return

        sections_to_add_citations_to = ["Introduction", "Related_Work", "Discussion"]

        for section_name in sections_to_add_citations_to:
            if section_name in self.generated_sections:
                print(f"Attempting to add/embed citations in: {section_name}...")
                try:
                    original_section_content = self.generated_sections[section_name]

                    if "embed_citation_prompt" not in self.prompts:
                        print(f"[WARNING] 'embed_citation_prompt' not found in prompts. Cannot add citations to {section_name}.")
                        continue
                    
                    # This prompt should ask for text/markdown with \\cite{}
                    citation_embedding_prompt = self.prompts.embed_citation_prompt.format(
                        section=section_name,
                        section_content=original_section_content, 
                        references=reference_list_str_for_embedding_prompt, 
                    )

                    section_with_citation_placeholders, _ = get_response_from_llm(
                        msg=citation_embedding_prompt,
                        client=self.client,
                        model=self.model,
                        system_message=self.prompts.citation_system_prompt, 
                        temperature=0.2 
                    )
                    
                    def replace_embedded_citation_placeholder(match_obj):
                        placeholder_text = match_obj.group(1).strip()
                        ref_key_match = re.match(r"Ref Key:\\s*(\\S+)", placeholder_text, re.IGNORECASE)
                        if ref_key_match:
                            potential_key = ref_key_match.group(1).lower()
                            if potential_key in self.title_to_bibtex_key_map: 
                                return f"\\\\cite{{{self.title_to_bibtex_key_map[potential_key]}}}"
                        
                        normalized_placeholder = placeholder_text.lower()
                        if normalized_placeholder in self.title_to_bibtex_key_map:
                            return f"\\\\cite{{{self.title_to_bibtex_key_map[normalized_placeholder]}}}"
                        
                        if placeholder_text in self.title_to_bibtex_key_map.values():
                            return f"\\\\cite{{{placeholder_text}}}"

                        for title_from_map, key_from_map in self.title_to_bibtex_key_map.items():
                             if placeholder_text.lower() in title_from_map or title_from_map in placeholder_text.lower():
                                 print(f"[INFO] Heuristic citation match (embed) for '{placeholder_text}' -> '{key_from_map}'")
                                 return f"\\\\cite{{{key_from_map}}}"

                        print(f"[WARNING] Could not map embedded citation placeholder '{placeholder_text}' in {section_name}.")
                        return f"\\\\cite{{{placeholder_text}}} % FIXME: Unmapped embedded citation"

                    final_section_with_citations = re.sub(r"\\\\cite\\{(.+?)\\}", replace_embedded_citation_placeholder, section_with_citation_placeholders)

                    if len(final_section_with_citations) > 0.7 * len(original_section_content):
                         self.generated_sections[section_name] = final_section_with_citations
                         original_cites = len(re.findall(r"\\\\cite\\{.*?\\}", original_section_content))
                         new_cites = len(re.findall(r"\\\\cite\\{.*?\\}", final_section_with_citations))
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