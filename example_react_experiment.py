#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

from tiny_scientist import TinyScientist


def main():
    parser = argparse.ArgumentParser(description="Run experiments using TinyScientist with ReAct")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o", 
        help="Specify the LLM model to use (e.g., gpt-4o, claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        default="general", 
        choices=["chemistry", "physics", "general"],
        help="Specify the experiment domain: chemistry, physics, or general"
    )
    parser.add_argument(
        "--intent", 
        type=str, 
        default=None,
        help="Experiment intent description (e.g., 'Chemistry experiment: measure NaCl solubility in different solvents')"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output", 
        help="Output directory path"
    )
    parser.add_argument(
        "--max-iterations", 
        type=int, 
        default=10, 
        help="Maximum number of ReAct iterations"
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="acl", 
        choices=["acl", "iclr"],
        help="Paper template format"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Default experiment intent (if not provided)
    if args.intent is None:
        if args.domain == "chemistry":
            args.intent = "Investigate NaCl solubility in water and ethanol at different temperatures"
        elif args.domain == "physics":
            args.intent = "Compare the relationship between thermal conductivity and electrical resistivity of different materials"
        else:
            args.intent = "Explore the performance of simple machine learning models on small datasets"
    
    print(f"🚀 Using {args.model} model for a {args.domain} experiment: {args.intent}")
    
    # Initialize TinyScientist
    scientist = TinyScientist(
        model=args.model,
        output_dir=output_dir,
        template=args.template
    )
    
    # Step 1: Generate research idea
    idea = scientist.think(intent=args.intent)
    
    # Step 2: Execute experiment using react_experiment method
    status, experiment_dir = scientist.react_experiment(
        idea=idea, 
        domain=args.domain,
        max_iterations=args.max_iterations
    )
    
    # If experiment successful, generate research paper
    if status:
        # Step 3: Write paper
        pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)
        
        # Step 4: Review paper (optional)
        review = scientist.review(pdf_path=pdf_path)
        
        print(f"✅ Experiment workflow complete!")
        print(f"📄 Paper saved at: {pdf_path}")
    else:
        print("❌ Experiment failed to complete successfully. Cannot generate paper.")


if __name__ == "__main__":
    main() 