#!/usr/bin/env python3
import argparse
import json

from code_searcher import CodeSearcher  # Ensure CodeSearcher is properly implemented


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for GitHub repositories or code snippets.")

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query for GitHub repositories or code"
    )
    parser.add_argument(
        "--result-limit",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["repositories", "code"],
        default="repositories",
        help="Type of GitHub search: repositories or code"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save retrieved search results as JSON"
    )
    return parser.parse_args()

def main() -> int:
    args: argparse.Namespace = parse_args()

    try:
        # Initialize CodeSearcher instance
        searcher = CodeSearcher()
        print(f"Searching for {args.search_type} on GitHub...")

        if args.search_type == "repositories":
            results = searcher.search_github_repositories(query=args.query, result_limit=args.result_limit)
        else:
            results = searcher.search_github_code(query=args.query, result_limit=args.result_limit)

        if not results:
            print("No results found.")
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
    exit(main())
