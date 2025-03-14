#!/usr/bin/env python3
import argparse
import json

from tiny_scientist.tool import PaperSearchTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for academic papers.")

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query for retrieving academic papers"
    )
    parser.add_argument(
        "--result-limit",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["semanticscholar", "openalex"],
        default="semanticscholar",
        help="Search engine for retrieving papers"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save retrieved papers as JSON"
    )
    return parser.parse_args()

def main() -> int:
    args: argparse.Namespace = parse_args()

    try:
        # Initialize PaperSearchTool instance
        searcher = PaperSearchTool()
        print(f"Searching for papers using {args.engine} engine...")

        papers = searcher.search_for_papers(
            query=args.query, result_limit=args.result_limit, engine=args.engine
        )
        results = searcher.format_github_results(papers)

        if not results:
            print("No papers found.")
            return 1

        # Display results
        print(json.dumps(results, indent=4))

        # Save results if output path is provided
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    print('rahs')
    exit(main())
