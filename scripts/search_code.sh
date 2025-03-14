#!/bin/bash

# Define default parameters
QUERY="machine learning"
RESULT_LIMIT=10
SEARCH_TYPE="repositories"  # Change to "code" for searching code snippets
OUTPUT="github_results.json"

# Run the CodeSearchTool script
python3 search_code.py --query "$QUERY" --result-limit $RESULT_LIMIT --search-type "$SEARCH_TYPE" --output "$OUTPUT"
