from __future__ import annotations

import abc
import json
import os
import re
import sys
from contextlib import redirect_stdout
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import toml
from fastmcp import FastMCP  # type: ignore
from rich import print

import tiny_scientist
from tiny_scientist.budget_checker import BudgetChecker
from tiny_scientist.configs import Config
from tiny_scientist.utils.error_handler import api_calling_error_exponential_backoff
from tiny_scientist.utils.llm import create_client, get_response_from_llm

PACKAGE_ROOT = Path(tiny_scientist.__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config.toml"
config = toml.load(CONFIG_PATH) if CONFIG_PATH.exists() else {"core": {}}


class BaseTool(abc.ABC):
    def __init__(self, cost_tracker: Optional[BudgetChecker] = None) -> None:
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.github_token = config["core"].get("github_token", None)

    @abc.abstractmethod
    def run(
        self, query: str, result_limit: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        pass


class DrawerTool(BaseTool):
    def __init__(
        self,
        model: Any,
        prompt_template_dir: Optional[str] = None,
        temperature: float = 0.75,
    ):
        super().__init__()
        self.client, self.model = create_client(model)
        self.temperature = temperature

        # Load prompt templates using Config
        self.config = Config(prompt_template_dir)
        self.prompts = self.config.prompt_template.drawer_prompt

        def escape_curly_braces(text: str) -> str:
            return re.sub(r"({|})", r"{{\1}}", text)

        def extract_pdf_text_from_resource(package: str, filename: str) -> str:
            with resources.files(package).joinpath(filename).open("rb") as f:
                doc = fitz.open(stream=f.read(), filetype="pdf")
                extracted = [page.get_text().strip() for page in doc]
                return "\n\n".join(extracted)

        method_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "framework.pdf"
        )
        result_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "result.pdf"
        )

        method_sample = escape_curly_braces(method_sample_raw)
        result_sample = escape_curly_braces(result_sample_raw)

        self.system_prompts = self.prompts.diagram_system_prompt.format(
            method_sample=method_sample,
            result_sample=result_sample,
        )

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        try:
            query_dict = json.loads(query)
            section_name = query_dict.get("section_name")
            section_content = query_dict.get("section_content")
        except (json.JSONDecodeError, TypeError, AttributeError):
            raise ValueError(
                "Expected query to be a JSON string with 'section_name' and 'section_content'."
            )

        diagram = self.draw_diagram(
            section_name=section_name, section_content=section_content
        )

        results = {}
        if diagram:
            results["diagram"] = {
                "summary": diagram.get("summary", ""),
                "svg": diagram.get("svg", ""),
            }
        self.cost_tracker.report()
        return results

    def draw_diagram(
        self,
        section_name: str,
        section_content: str,
        msg_history: Optional[List[Dict[str, Any]]] = None,
        return_msg_history: bool = False,
    ) -> Any:
        # Use default system prompt if none provided
        section_prompt = self._get_section_prompts(section_name, section_content)

        diagram, updated_msg_history = self._generate_diagram(
            section_prompt, self.system_prompts, msg_history
        )

        return (diagram, updated_msg_history) if return_msg_history else diagram

    def _get_section_prompts(self, section_name: str, section_text: str) -> str:
        section_prompt = self.prompts.section_prompt[section_name].format(
            section_text=section_text
        )

        return section_prompt

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_diagram(
        self,
        section_prompt: str,
        drawer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Ensure msg_history is a list
        msg_history = msg_history or []

        # Generate diagram
        llm_response, msg_history = get_response_from_llm(
            section_prompt,
            model=self.model,
            client=self.client,
            system_message=drawer_system_prompt,
            msg_history=msg_history,
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="generate_diagram",
        )

        diagram = self._extract_diagram(llm_response)
        return diagram, msg_history

    def _extract_diagram(self, response: str) -> Dict[str, Any]:
        result = {"summary": "", "svg": "", "full_response": response}

        try:
            parsed = json.loads(response)
            summary = parsed["summary"]
            svg = parsed["svg"]
        except json.JSONDecodeError:
            svg_match = re.search(r"<svg.*?</svg>", response, re.DOTALL)
            svg = svg_match.group(0) if svg_match else ""
            summary = (
                re.sub(r"<svg.*?</svg>", "", response, flags=re.DOTALL)
                .strip()
                .split("\n")[0]
            )

        if "<svg" in svg and "</svg>" in svg:
            result["summary"] = summary
            result["svg"] = self._clean_svg(svg)
        else:
            print("[ERROR] SVG missing or too short.")
        return result

    def _clean_svg(self, svg: str) -> str:
        # Strip any outer code block delimiters
        svg = svg.strip()
        svg = re.sub(r"^```(?:svg)?", "", svg)
        svg = re.sub(r"```$", "", svg)

        # Replace problematic ampersands
        svg = svg.replace("&", "&amp;")

        # Ensure no double XML declarations
        svg = re.sub(r"<\?xml.*?\?>", "", svg, count=1)

        # Remove extra whitespace lines
        svg = "\n".join([line for line in svg.splitlines() if line.strip()])

        return svg.strip()


app = FastMCP("tiny-scientist-drawer", description="Diagram generation MCP server")


@app.tool(name="drawer.run", description="Generate diagram SVG and summary")
def run_drawer(
    query: str,
    model: str,
    prompt_template_dir: Optional[str] = None,
    temperature: float = 0.75,
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        tool = DrawerTool(
            model=model,
            prompt_template_dir=prompt_template_dir,
            temperature=temperature,
        )
        return tool.run(query)


if __name__ == "__main__":
    app.run()
