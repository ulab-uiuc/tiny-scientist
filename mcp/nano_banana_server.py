#!/usr/bin/env python3
"""
Nano Banana MCP Server - Simple Gemini-powered diagram generation
Uses gemini-2.5-flash-image-preview to generate scientific diagrams
"""

import os

import google.genai as genai
from fastmcp import FastMCP

# Create FastMCP server
mcp = FastMCP(name="Nano Banana Diagram Generator")


def _png_to_svg_embed(png_base64: str, width: int = 600, height: int = 400) -> str:
    """Convert PNG base64 to SVG with embedded image."""
    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <image href="data:image/png;base64,{png_base64}" width="{width}" height="{height}"/>
</svg>"""


@mcp.tool()
def generate_diagram(section_name: str, section_content: str) -> dict[str, str]:
    """
    Generate a scientific diagram for a given section using Gemini.

    Args:
        section_name: Type of section (Introduction, Method, Experimental_Setup, Results)
        section_content: The text content of the section

    Returns:
        Dictionary with 'summary' and 'svg' keys (SVG contains embedded PNG)
    """
    try:
        # Check for API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"summary": "Error: GEMINI_API_KEY not set", "svg": ""}

        # Configure Gemini client
        client = genai.Client(api_key=api_key)

        # Create diagram generation prompt
        prompt = f"""Create a professional scientific diagram for the {section_name} section.

Content: {section_content}

Generate a clear, informative diagram that visualizes the key concepts from this {section_name.lower()} section.
Use scientific illustration style with proper labels and clear visual hierarchy."""

        # Generate image with Gemini 2.0 Flash
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt + "\n\nPlease generate this as a visual diagram/chart.",
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
            ),
        )

        # Check if response contains images
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    # Extract PNG data and convert to SVG
                    png_data = part.inline_data.data
                    svg_content = _png_to_svg_embed(png_data)
                    summary = (
                        f"Generated {section_name} diagram using Gemini Nano Banana"
                    )

                    return {"summary": summary, "svg": svg_content}

        # Fallback if no image generated
        summary = f"Generated {section_name} diagram using Gemini (text response)"
        fallback_text = response.text if response.text else "No content generated"

        fallback_svg = f"""<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
    <text x="300" y="50" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#212529">
        {section_name} Diagram (Nano Banana)
    </text>
    <text x="300" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#6c757d">
        Generated with Gemini-2.0-Flash
    </text>
    <rect x="50" y="150" width="500" height="200" fill="#e9ecef" stroke="#adb5bd" stroke-width="1" rx="10"/>
    <text x="300" y="180" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#495057">
        Section: {section_name}
    </text>
    <text x="300" y="210" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#6c757d">
        Content Length: {len(section_content)} characters
    </text>
    <foreignObject x="60" y="220" width="480" height="120">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-size: 10px; color: #868e96; word-wrap: break-word; overflow: hidden;">
            {fallback_text[:200]}{"..." if len(fallback_text) > 200 else ""}
        </div>
    </foreignObject>
</svg>"""

        return {"summary": summary, "svg": fallback_svg}

    except Exception as e:
        return {"summary": f"Error generating diagram: {str(e)}", "svg": ""}


if __name__ == "__main__":
    print("[Nano Banana] Starting Gemini-powered MCP server...")
    mcp.run()
