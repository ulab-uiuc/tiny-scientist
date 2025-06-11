#!/bin/bash

# Define default parameters
QUERY="deep learning in healthcare"
RESULT_LIMIT=10
ENGINE="semanticscholar"
OUTPUT="papers.json"

# Run the PaperSearchTool script
python3 ../examples/search_paper.py --query "$QUERY" --result-limit $RESULT_LIMIT --engine "$ENGINE" --output "$OUTPUT"
