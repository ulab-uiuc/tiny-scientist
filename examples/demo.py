import argparse
import os

from tiny_scientist.scientist import TinyScientist


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyScientist pipeline.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/demo_3",
        help="Base output directory",
    )
    parser.add_argument(
        "--prompt_template_dir",
        type=str,
        default=None,
        help="Configuration directory with prompt YAML files",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="LLM model to use"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="acl",
        help="Paper format template (e.g. acl, iclr)",
    )
    parser.add_argument(
        "--enable_safety_check",
        type=bool,
        default=True,
        help="Enable safety check for input prompts (True/False)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate TinyScientist and run pipeline
    scientist = TinyScientist(
        model=args.model,
        output_dir=args.output_dir,
        prompt_template_dir=args.prompt_template_dir,
        template=args.template,
        enable_safety_check=args.enable_safety_check,
    )

    idea = scientist.think(
        intent="latent reasoning for large language models"
    )

    if isinstance(idea, list):
        idea = idea[0]

    idea["is_experimental"] = False
    pdf_path = scientist.write(idea=idea, experiment_dir=args.output_dir)
    scientist.review(pdf_path=pdf_path)


if __name__ == "__main__":
    main()
