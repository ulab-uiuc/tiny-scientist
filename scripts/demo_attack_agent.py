#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from tiny_scientist import TinyScientist


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TinyScientist with malicious agents")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--intent",
        type=str,
        default="Develop a machine learning algorithm for analyzing social media data",
        help="Research intent",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help="Domain for the research (empty for general)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output",
    )
    parser.add_argument(
        "--attack_probability",
        type=float,
        default=0.8,
        help="Probability of attack occurring (0-1)",
    )
    parser.add_argument(
        "--attack_severity",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Severity of the attack",
    )
    parser.add_argument(
        "--comparative",
        action="store_true",
        help="Run both normal and attack versions for comparison",
    )
    
    return parser.parse_args()


def main():
    """Run the demo with malicious agents."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Display configuration
    print("=== TinyScientist with Malicious Agents Demo ===")
    print(f"Model: {args.model}")
    print(f"Research Intent: {args.intent}")
    print(f"Domain: {args.domain or '(general)'}")
    print(f"Attack Probability: {args.attack_probability}")
    print(f"Attack Severity: {args.attack_severity}")
    print(f"Output Directory: {args.output_dir}")
    print("="*50)
    
    if args.comparative:
        print("\n=== Running NORMAL Version (No Attacks) ===\n")
        
        # Create normal instance
        normal_scientist = TinyScientist(
            model=args.model,
            output_dir=f"{args.output_dir}/normal",
            enable_malicious_agents=False,
        )
        
        # Generate normal idea
        normal_result = normal_scientist.think(
            intent=args.intent,
            domain=args.domain,
        )
        
        print("\n=== Running ATTACK Version ===\n")
    
    # Create scientist with malicious agents enabled
    attack_scientist = TinyScientist(
        model=args.model,
        output_dir=f"{args.output_dir}/attack" if args.comparative else args.output_dir,
        enable_malicious_agents=True,
        attack_probability=args.attack_probability,
        attack_severity=args.attack_severity,
    )
    
    # Generate idea with potential attacks
    attack_result = attack_scientist.think(
        intent=args.intent,
        domain=args.domain,
    )
    
    # Optionally compare results
    if args.comparative:
        print("\n=== COMPARISON ===\n")
        print("Normal Result:")
        print(f"Title: {normal_result.get('Title', 'N/A')}")
        print(f"Score: {normal_result.get('Score', 'N/A')}")
        
        print("\nAttack Result:")
        print(f"Title: {attack_result.get('Title', 'N/A')}")
        print(f"Score: {attack_result.get('Score', 'N/A')}")
        
        print("\nCheck the output directories for full details.")


if __name__ == "__main__":
    main() 