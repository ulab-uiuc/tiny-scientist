import json
from typing import Any, Optional

import pymupdf
import pymupdf4llm
from pypdf import PdfReader


def load_paper(pdf_path: str, num_pages: Optional[int] = None, min_size: int = 100) -> Any:
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
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")
    return text

def load_review(review_path: str) -> Any:
    with open(review_path, "r") as f:
        return json.load(f)["review"]
