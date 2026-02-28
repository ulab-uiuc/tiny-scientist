#!/usr/bin/env python3
import argparse
import json
import os

import _bootstrap
from tiny_scientist.tool_impls import DrawerTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diagrams from text using an LLM."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Text content to generate a diagram from",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a text file to generate a diagram from",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use (default: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagram_output.json",
        help="Path to save the generated diagram as JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Temperature for LLM generation (default: 0.75)",
    )
    return parser.parse_args()


def main() -> int:
    args: argparse.Namespace = parse_args()

    if not args.text and not args.input_file:
        print("Error: Either --text or --input-file must be provided")
        return 1

    try:
        # Get text content from file or command line argument
        if args.input_file:
            with open(args.input_file, "r") as f:
                text = f.read()
        else:
            text = args.text

        # Get prompt templates directory
        current_dir = os.path.dirname(os.path.realpath(__file__))
        prompt_template_dir = os.path.join(
            os.path.dirname(current_dir), "tiny_scientist", "prompts"
        )

        # Initialize DrawerTool
        drawer = DrawerTool(
            model=args.model,
            prompt_template_dir=prompt_template_dir,
            temperature=args.temperature,
        )

        print(f"Generating diagram using {args.model}...")

        query = json.dumps(
            {
                "section_name": "Method",
                "section_content": text,
            }
        )
        result = drawer.run(query)
        diagram = result.get("diagram", {})

        if not diagram or not diagram.get("svg"):
            print("Failed to generate diagram.")
            return 1

        # Display summary
        if diagram.get("summary"):
            print("\nDiagram Summary:")
            print(diagram["summary"])

        # Save results
        output_data = {
            "summary": diagram.get("summary", ""),
            "svg": diagram.get("svg", ""),
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"\nDiagram saved to {args.output}")

        # Also save SVG directly if available
        if diagram.get("svg"):
            svg_path = args.output.replace(".json", ".svg")
            with open(svg_path, "w") as f:
                f.write(diagram["svg"])
            print(f"SVG file saved to {svg_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
