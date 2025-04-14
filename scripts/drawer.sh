#!/bin/bash

# Define default parameters
MODEL="gpt-4o-2024-08-06"
OUTPUT="diagram_output.json"
TEMPERATURE=0.75

# Check if input text or file is provided
if [ "$1" == "" ]; then
  echo "Usage: ./drawer.sh <text or path to text file>"
  echo "Example: ./drawer.sh \"Design a system for text to image generation using LLMs\""
  echo "Example: ./drawer.sh --input-file path/to/text_file.txt"
  exit 1
fi

# Determine if the input is a file or text
if [[ "$1" == "--input-file" ]]; then
  # Run the DrawerTool script with an input file
  python3 drawer.py --input-file "$2" --model "$MODEL" --output "$OUTPUT" --temperature "$TEMPERATURE"
else
  # Run the DrawerTool script with text input
  python3 drawer.py --text "$*" --model "$MODEL" --output "$OUTPUT" --temperature "$TEMPERATURE"
fi
