#!/usr/bin/env python3
import argparse
import json
import os

from tiny_scientist.coder import Coder  # Ensure Coder is properly implemented
from tiny_scientist.llm import AVAILABLE_LLMS, create_client
from tiny_scientist.writer import Writer

def parse_args():
    parser = argparse.ArgumentParser(description="Write paper.")
    
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to the experiment directory containing experiment details"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        choices=AVAILABLE_LLMS,
        help="Model to use for writing and refinement"
    )
    parser.add_argument(
        "--num-cite-rounds",
        type=int,
        default=2,
        help="Number of citation addition rounds"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["semanticscholar", "openalex"],
        default="semanticscholar",
        help="Search engine for citation retrieval"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save final paper PDF (defaults to experiment directory)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Create LLM client and model
        client, model = create_client(args.model)
        
        # Ensure Coder instance exists (if required)
        coder = Coder(base_dir=args.experiment, model=model)
        print("Coder instance created.")
        # Initialize Writer
        writer = Writer(
            model=model,
            client=client,
            base_dir=args.experiment,
            coder=coder
        )
        # idea = {
        #     "Name": "Generated_Paper",
        #     "Title": prompt_data.get("task_description", "Research Paper"),
        # }
        
        # idea should be import from args.experiemnt and idea.json
        with open(os.path.join(args.experiment, "idea.json"), "r") as f:
            idea = json.load(f)
    
        # Perform paper writing
        print("\nStarting paper write-up...")
        writer.perform_writeup(
            idea=idea,
            folder_name=args.experiment,
            num_cite_rounds=args.num_cite_rounds,
            engine=args.engine
        )
        
        # Save output PDF
        output_path = args.output or os.path.join(args.experiment, "Generated_Paper.pdf")
        print(f"\nFinal paper saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
