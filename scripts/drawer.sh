#!/bin/bash

# Define default parameters
MODEL="gpt-4o-mini"
OUTPUT="diagram_output.json"
TEMPERATURE=0.75
TMP_INPUT="example_input.txt"

# If no arguments are given, run with built-in example
if [ "$#" -eq 0 ]; then
  echo "[INFO] No input provided. Running with built-in example..."

  # Create a temporary input file with sample text
  cat <<EOF > $TMP_INPUT
We propose a three-stage pipeline for document classification using large language models. First, the input document is segmented into paragraphs and tokenized. Each segment is independently passed through a pretrained transformer encoder (e.g., BERT) to generate contextual embeddings. Second, a hierarchical attention mechanism aggregates segment-level embeddings into a global document representation, emphasizing the most informative segments.
Finally, the aggregated document representation is fed into a feed-forward classification head that outputs the final label.
The model is trained end-to-end using cross-entropy loss and optimized with AdamW. This architecture allows for flexible handling of long documents and improves interpretability by localizing important content through attention weights.
EOF

  # Run the drawer script with the sample input
  python3 ../examples/drawer.py --input-file $TMP_INPUT --model "$MODEL" --output "$OUTPUT" --temperature "$TEMPERATURE"

  # Clean up
  rm -f $TMP_INPUT
  exit 0
fi

# Custom input provided by user
if [[ "$1" == "--input-file" ]]; then
  python3 ../examples/drawer.py --input-file "$2" --model "$MODEL" --output "$OUTPUT" --temperature "$TEMPERATURE"
else
  python3 ../examples/drawer.py --text "$*" --model "$MODEL" --output "$OUTPUT" --temperature "$TEMPERATURE"
fi
