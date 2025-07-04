import json
import os
import re
from importlib import resources
from typing import Any, Dict, Optional

import fitz
import httpx
import toml
from mcp.server.fastmcp import FastMCP

from tiny_scientist.configs import Config

# Initialize FastMCP server
mcp = FastMCP("drawer")

# Load config
config_path = os.path.join(os.path.dirname(__file__), "../..", "config.toml")
config = toml.load(config_path) if os.path.exists(config_path) else {"core": {}}

# LLM configuration
LLM_MODEL = config["core"].get("model", "gpt-4o-mini")
LLM_API_KEY = config["core"].get("llm_api_key", "")
LLM_TEMPERATURE = config["core"].get("temperature", 0.75)

# Load prompt templates from the configs module
prompt_config = Config()
prompts = prompt_config.prompt_template.drawer_prompt


def escape_curly_braces(text: str) -> str:
    """Escape curly braces in text to prevent format string issues."""
    return re.sub(r"({|})", r"{{\1}}", text)


def extract_pdf_text_from_resource(package: str, filename: str) -> str:
    """Extract text from a PDF resource file."""
    with resources.files(package).joinpath(filename).open("rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        extracted = [page.get_text().strip() for page in doc]
        return "\n\n".join(extracted)


def get_section_prompts(section_name: str, section_text: str) -> str:
    """Get section-specific prompts."""
    section_prompt = prompts.section_prompt[section_name].format(
        section_text=section_text
    )
    return section_prompt


async def make_llm_request(prompt: str, system_message: str) -> Optional[str]:
    """Make a request to the LLM API."""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else None
        except Exception as e:
            print(f"LLM API request failed: {e}")
            return None


def extract_diagram_data(response: str) -> Dict[str, Any]:
    """Extract diagram data from LLM response."""
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
        result["svg"] = clean_svg(svg)
    else:
        print("[ERROR] SVG missing or too short.")
    return result


def clean_svg(svg: str) -> str:
    """Clean and format SVG content."""
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


# Initialize system prompt with sample data
def initialize_system_prompt() -> str:
    """Initialize the system prompt with sample data."""
    try:
        method_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "framework.pdf"
        )
        result_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "result.pdf"
        )

        method_sample = escape_curly_braces(method_sample_raw)
        result_sample = escape_curly_braces(result_sample_raw)

        return prompts.diagram_system_prompt.format(
            method_sample=method_sample,
            result_sample=result_sample,
        )
    except Exception as e:
        print(f"[WARNING] Failed to load sample data: {e}")
        return "You are a diagram generation assistant. Generate SVG diagrams based on research paper sections."


SYSTEM_PROMPT = initialize_system_prompt()


@mcp.tool()
async def generate_diagram(section_name: str, section_content: str) -> str:
    """Generate an SVG diagram for a research paper section.

    Args:
        section_name: Name of the paper section (e.g., "Method", "Results")
        section_content: Content of the section to visualize
    """
    print(f"[Drawer] Generating diagram for section: {section_name}")

    if not section_content.strip():
        return json.dumps({"error": "Section content cannot be empty"})

    # Get section-specific prompts
    section_prompt = get_section_prompts(section_name, section_content)

    # Generate diagram using LLM
    llm_response = await make_llm_request(section_prompt, SYSTEM_PROMPT)

    if not llm_response:
        return json.dumps({"error": "Failed to generate diagram from LLM"})

    # Extract diagram data
    diagram = extract_diagram_data(llm_response)

    # Format response
    result = {
        "diagram": {
            "summary": diagram.get("summary", ""),
            "svg": diagram.get("svg", ""),
        }
    }

    return json.dumps(result, indent=2)


@mcp.tool()
async def validate_svg(svg_content: str) -> str:
    """Validate and clean SVG content.

    Args:
        svg_content: SVG content to validate and clean
    """
    print("[Drawer] Validating and cleaning SVG content")

    if not svg_content.strip():
        return json.dumps({"error": "SVG content cannot be empty"})

    try:
        cleaned_svg = clean_svg(svg_content)

        # Basic validation - check if it looks like valid SVG
        if "<svg" in cleaned_svg and "</svg>" in cleaned_svg:
            result = {
                "valid": True,
                "cleaned_svg": cleaned_svg,
                "message": "SVG is valid and has been cleaned",
            }
        else:
            result = {
                "valid": False,
                "cleaned_svg": "",
                "message": "SVG appears to be invalid or incomplete",
            }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "cleaned_svg": "",
                "message": f"Error validating SVG: {str(e)}",
            }
        )


@mcp.tool()
async def get_supported_sections() -> str:
    """Get list of supported section types for diagram generation."""
    supported_sections = list(prompts.section_prompt.keys())

    result = {
        "supported_sections": supported_sections,
        "description": "These are the section types that have specialized prompts for diagram generation",
    }

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
