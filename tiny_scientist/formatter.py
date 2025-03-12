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


class Formatter:
    def __init__(
        self,
        template: str,
        model: str,
        client: Any,
        s2_api_key: Optional[str] = None,
    ):
        self.template = template
        self.model = model
        self.client = client
        self.s2_api_key = s2_api_key

    # base class for setting up the template
    def _set_template(self, template: str) -> None:
        if template is not None:
            script_dir = osp.dirname(__file__)
            project_root = osp.abspath(osp.join(script_dir, ".."))
            source_template_dir = osp.join(project_root, "tiny_scientist", f"{template}_latex")

            if osp.isdir(source_template_dir):
                dest_template_dir = osp.join(self.base_dir, "latex")

                if osp.exists(dest_template_dir):
                    shutil.rmtree(dest_template_dir)
                shutil.copytree(source_template_dir, dest_template_dir)
        
            self.dest_dir = dest_template_dir

        else:
            pass

    def _save_bib(self, bib_list: List[str], dest_dir: str) -> None:
        if self.template == 'acl':
            bib_path = osp.join(dest_dir, "latex", "custom.bib")
        elif self.template == 'iclr':
            bib_path = osp.join(dest_dir, "custom.bib")

        if osp.exists(bib_path):
            with open(bib_path, "r", encoding="utf-8") as f:
                existing_bib = f.read()
        else:
            existing_bib = ""

        existing_entries = set(existing_bib.split("\n\n")) if existing_bib else set()

        combined_entries = existing_entries.union(bib_list)
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_entries))

    