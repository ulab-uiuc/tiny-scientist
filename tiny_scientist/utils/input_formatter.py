import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
import pymupdf4llm
from pypdf import PdfReader
from rich import print


class InputFormatter:
    def _load_paper(
        self, pdf_path: str, num_pages: Optional[int] = None, min_size: int = 100
    ) -> str:
        """
        Loads a PDF, attempting to convert it to Markdown via pymupdf4llm.
        If that fails, falls back to direct pymupdf extraction, and then
        finally to PyPDF2. Returns the extracted text as a single string.
        """
        try:
            if num_pages is None:
                text = pymupdf4llm.to_markdown(pdf_path)
            else:
                reader = PdfReader(pdf_path)
                min_pages = min(len(reader.pages), num_pages)
                text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
            try:
                doc = pymupdf.open(pdf_path)
                if num_pages:
                    doc = doc[:num_pages]
                text = ""
                for page in doc:
                    text += page.get_text()
                if len(text) < min_size:
                    raise Exception("Text too short")
            except Exception as e:
                print(f"Error with pymupdf, falling back to PyPDF2: {e}")
                reader = PdfReader(pdf_path)
                if num_pages is None:
                    text = "".join(page.extract_text() for page in reader.pages)
                else:
                    text = "".join(
                        page.extract_text() for page in reader.pages[:num_pages]
                    )
                if len(text) < min_size:
                    raise Exception("Text too short")
        return str(text)

    def _extract_subsections(
        self, section_text: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Helper function to parse sub-subsections of the form:
        '**x.x** **Subsection Title**'.
        Returns a tuple (clean_text, subsections), where 'clean_text' is the
        remaining text outside these subsections, and 'subsections' is a list
        of dicts with the keys 'subsection_number', 'subsection_title', and
        'subsection_content'.
        """
        subsections = []
        subsec_pattern = re.compile(
            r"(?m)^\*\*(\d+\.\d+)\*\*\s+\*\*(.*?)\*\*\s*(.*?)(?=^\*\*\d+\.\d+\*\*|\Z)",
            re.DOTALL,
        )
        matches = list(subsec_pattern.finditer(section_text))

        if not matches:
            return section_text.strip(), []

        leftover_parts = []
        last_end = 0

        for m in matches:
            start_idx = m.start()
            leftover = section_text[last_end:start_idx]
            leftover_parts.append(leftover)

            subsection_number = m.group(1).strip()
            subsection_title = m.group(2).strip()
            subsection_content = m.group(3).strip()

            subsections.append(
                {
                    "subsection_number": subsection_number,
                    "subsection_title": subsection_title,
                    "subsection_content": subsection_content,
                }
            )

            last_end = m.end()

        leftover_parts.append(section_text[last_end:])
        clean_text = "\n".join(part.strip() for part in leftover_parts).strip()

        return clean_text, subsections

    def _parse_markdown(self, markdown_str: str) -> Dict[str, Any]:
        """
        Parses a markdown document with the following structure:

        1) Optional document title of the form:
           ## My Document Title

        2) Everything before '### Abstract' goes into 'header'.

        3) From '### Abstract' onward, each '### Some Heading' is treated
           as a top-level section. That section's content is everything
           until the next '### ' heading or the end of the document.

        4) Within each top-level section, sub-sections appear in lines
           of the form:
             **x.x** **Subsection Title**
           and continue until the next sub-section or the next top-level section.

        Returns a dictionary of the form:

            {
              "title": "...",
              "header": "...",
              "sections": [
                {
                  "section_name": "...",
                  "content": "...",
                  "subsections": [
                    {
                      "subsection_number": "x.x",
                      "subsection_title": "...",
                      "subsection_content": "..."
                    },
                    ...
                  ]
                },
                ...
              ]
            }
        """
        # 1) Extract optional document title
        title_pattern = re.compile(r"(?m)^##\s+(.*)")
        title_match = title_pattern.search(markdown_str)
        title = ""
        if title_match:
            title = title_match.group(1).strip()
            full_line = title_match.group(0)
            markdown_str = markdown_str.replace(full_line, "", 1)

        # 2) Split out "header" from everything after '### Abstract'
        split_pattern = r"(?s)(.*?)^### Abstract(.*)"
        match = re.search(split_pattern, markdown_str, re.MULTILINE)

        if not match:
            return {"title": title, "header": markdown_str.strip(), "sections": []}

        part_before = match.group(1)
        part_after = "### Abstract" + match.group(2)
        header = part_before.strip()

        # 3) Extract top-level sections from 'part_after'
        section_pattern = re.compile(
            r"(?m)^###\s+(.*?)\s*\n" r"(.*?)(?=^###\s+|\Z)", re.DOTALL
        )

        raw_sections = section_pattern.findall(part_after)
        sections = []

        # Parse each top-level section
        for section_name, section_text in raw_sections:
            section_name = section_name.strip()
            clean_text, subsections_list = self._extract_subsections(section_text)
            section_dict = {
                "section_name": section_name,
                "content": clean_text,
                "subsections": subsections_list,
            }
            sections.append(section_dict)

        return {"title": title, "header": header, "sections": sections}

    def _load_review(self, review_path: str) -> str:
        """
        Loads a JSON file (at review_path) and returns the string under the 'review' key.
        The JSON is expected to have the structure: { "review": "..." }.
        """
        with open(review_path, "r", encoding="utf-8") as f:
            data: Dict[str, str] = json.load(f)
            return data["review"]

    def parse_paper_pdf_to_json(
        self, pdf_path: str, num_pages: Optional[int] = None, min_size: int = 100
    ) -> Dict[str, Any]:
        """
        Convenience method to load a PDF, convert it to text, parse the markdown,
        and return a structured JSON-like Python dictionary.

        If no sections are found during parsing, returns the raw PDF text in a
        compatible format with "title": "", "header": "", and a single section
        containing the full text.
        """
        pdf_text = self._load_paper(pdf_path, num_pages=num_pages, min_size=min_size)
        parsed_result = self._parse_markdown(pdf_text)

        # If no sections were found, return the raw text in a compatible format
        if not parsed_result.get("sections"):
            print("No sections found in parsed result, returning raw text")
            return {
                "title": "",
                "header": "",
                "sections": [
                    {
                        "section_name": "Full Text",
                        "content": pdf_text,
                        "subsections": [],
                    }
                ],
            }

        return parsed_result

    def parse_review_json(self, review_path: str) -> Dict[str, Any]:
        """
        Convenience method to load a JSON 'review' file, then parse it using the
        same markdown rules, returning a structured JSON-like dictionary.
        """
        review_text = self._load_review(review_path)
        return self._parse_markdown(review_text)
