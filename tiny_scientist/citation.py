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

from .llm import extract_json_between_markers, get_response_from_llm
from .searcher import PaperSearcher

class Citation:
    def __init__(
        self,
        model: str,
        client: Any,
        s2_api_key: Optional[str] = None,
    ):
        self.model = model
        self.client = client
        self.s2_api_key = s2_api_key
        self.citation_list = []
        self.bib_list = []
        self.paper_searcher = PaperSearcher(s2_api_key)

        # Load prompts
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "cite_prompt.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    # save all finalized citations into the custom.bib file
    def save_citations(self, bib_list: List[str], dest_dir: str, template: str) -> None:
        if template == 'acl':
            bib_path = osp.join(dest_dir, "latex", "custom.bib")
        elif template == 'iclr':
            bib_path = osp.join(dest_dir, "custom.bib")

        if osp.exists(bib_path):
            with open(bib_path, "r", encoding="utf-8") as f:
                existing_bib = f.read()
        else:
            existing_bib = ""

        existing_entries = set(existing_bib.split("\n\n")) if existing_bib else set()

        combined_entries = existing_entries.union(bib_list)
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_entries) + "\n")



    def extract_citations(self, text: str) -> List[str]:
        citation_patterns = [
            r"\\cite\{([^}]+)\}",
            r"\\citep\{([^}]+)\}",
            r"\\citet\{([^}]+)\}"
        ]
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                keys = [key.strip() for key in m.split(",")]
                self.citation_list.extend(keys)
        return self.citation_list

    def _get_citation_prompt(
        self,
        draft: str,
    ) -> Tuple[Optional[List[Dict[str, str]]], bool]:
        try:
            prompt = self.prompts["citation_insert_prompt"].format(section=draft)
            text = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts["citation_system_prompt"],
            )
            return text
        
        except Exception as e:
            print(f"Error in citation generation: {e}")
            return None
        
    def process_citations(self, generated_sections: Dict[str, Any], template:str, dest_dir: str) -> Dict[str, Any]:
        for section, text in generated_sections.items():
            if section in ["Title", "Abstract"]:
                continue
        
            modified_section, _ = self._get_citation_prompt(text)
            citations = self.extract_citations(modified_section)
            
            for citation in citations:
                if citation not in self.citation_list:
                    self.citation_list.append(citation)

            generated_sections[section] = modified_section

        self.citation_list = list(set(self.citation_list))
        print(self.citation_list)
        for citation_id in self.citation_list:
            bib = self.paper_searcher._fetch_bibtex(citation_id)  # Get BibTeX
            
            if bib:
                self.bib_list.append(bib)
            else:
                self.citation_list.remove(citation_id)  # Remove if no valid BibTeX

        print(len(self.bib_list))
        self.save_citations(self.bib_list, dest_dir, template)
        # you need recheck the generated_sections to remove the non-cited papers since we have removed them from the citation_list
        # basically if a citation is not in the citation_list but appear in paper, then it should be removed from the generated_sections
        # for section, text in generated_sections.items():
        #     if section in ["Title", "Abstract"]:
        #         continue  
        
        #     updated_text, _ = self._remove_non_cited_references(text, self.citation_list)
        #     generated_sections[section] = updated_text
            
        return generated_sections
    
    def _remove_non_cited_references(self, text: str, citation_list: List[str]) -> str:
        prompt = self.prompts["citation_clean_prompt"].format(
            section=text,
            valid_citations=", ".join(citation_list)
        )

        try:
            cleaned_text = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts["citation_system_prompt"],
            )
            return cleaned_text

        except Exception as e:
            print(f"Error in citation removal: {e}")
            return text  # Fallback: return original text if LLM call fails
        
    def _get_semantic_scholar_paper_id(self, query: str) -> Optional[str]:
        """Search for a paper ID in Semantic Scholar. Handles rate limits and retries."""
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        max_retries = 10
        delay = 1  # Initial wait time (seconds)

        for attempt in range(max_retries):
            rsp = requests.get(
                search_url,
                headers={"X-API-KEY": self.s2_api_key} if self.s2_api_key else {},
                params={
                    "query": query,
                    "limit": 1,  # Get the most relevant paper
                    "fields": "paperId,title,authors,venue,year",
                },
            )

            if rsp.status_code == 200:
                results = rsp.json()
                print(f"Semantic Scholar search results: {results}")
                if "data" in results and results["data"]:
                    return results["data"][0]["paperId"]  # Return the correct paper ID
                return None  # No matches found

            elif rsp.status_code == 429:
                print(f"Rate limited on attempt {attempt + 1}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff

            else:
                print(f"Semantic Scholar search failed for query: {query}, Status Code: {rsp.status_code}")
                return None  # Stop retrying on non-429 errors

        print(f"Semantic Scholar API failed after multiple attempts for query: {query}")
        return None



    