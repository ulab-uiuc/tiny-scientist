from __future__ import annotations

import sys
from contextlib import redirect_stdout
from typing import Dict, Optional

from fastmcp import FastMCP  # type: ignore

from tiny_scientist.tool_impls import DrawerTool

app = FastMCP("tiny-scientist-drawer", description="Diagram generation MCP server")


@app.tool(name="drawer.run", description="Generate diagram SVG and summary")
def run_drawer(
    query: str,
    model: str,
    prompt_template_dir: Optional[str] = None,
    temperature: float = 0.75,
    backend: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Generate a diagram SVG and summary for a given section.

    Args:
        query: JSON string with 'section_name' and 'section_content' fields.
        model: LLM model identifier used for diagram generation.
        prompt_template_dir: Optional path to custom prompt template directory.
        temperature: Sampling temperature for LLM-based generation (ignored for
            nano-banana backend).
        backend: Diagram backend to use.  Supported values:
            - ``"llm_svg"`` (default) — ask the LLM to write an SVG directly.
            - ``"nano-banana"`` — use OpenAI image generation (requires
              ``OPENAI_API_KEY``; optionally set ``NANO_BANANA_MODEL``).
            When *None* the value of the ``DRAWER_BACKEND`` environment variable
            is used, falling back to ``"llm_svg"``.
    """
    with redirect_stdout(sys.stderr):
        tool = DrawerTool(
            model=model,
            prompt_template_dir=prompt_template_dir,
            temperature=temperature,
            backend=backend,
        )
        return tool.run(query)


if __name__ == "__main__":
    app.run()
