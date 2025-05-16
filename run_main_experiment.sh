#!/bin/bash

# Default values
DEFAULT_MODEL="gpt-4o"
DEFAULT_OUTPUT_DIR_BASE="./output/main_experiments_run"
DEFAULT_TEMPLATE="acl"
DEFAULT_INPUT_FILE="./data/ScienceSafetyData/Dataset/med.json" # Example default input
DEFAULT_OUTPUT_JSONL="./output/main_experiment_results.jsonl"

# Help function
show_help() {
    echo "Usage: run_main_experiment.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                   Show this help message and exit."
    echo "  --input-file FILE_PATH       Path to the input JSON file with tasks. (Default: ${DEFAULT_INPUT_FILE})"
    echo "  --output-file FILE_PATH      Path to the output JSONL file for results. (Default: ${DEFAULT_OUTPUT_JSONL})"
    echo "  --model MODEL_NAME           LLM model to use for TinyScientist. (Default: ${DEFAULT_MODEL})"
    echo "  --output-dir-base DIR_PATH   Base directory for TinyScientist task artifacts. (Default: ${DEFAULT_OUTPUT_DIR_BASE})"
    echo "  --template TEMPLATE_NAME     Paper template for writers (acl, iclr). (Default: ${DEFAULT_TEMPLATE})"
    echo ""
    echo "Example:"
    echo "  ./run_main_experiment.sh --input-file ./my_tasks.json --output-file ./my_results.jsonl"
}

# Initialize variables with default values
INPUT_FILE="${DEFAULT_INPUT_FILE}"
OUTPUT_FILE="${DEFAULT_OUTPUT_JSONL}"
MODEL="${DEFAULT_MODEL}"
OUTPUT_DIR_BASE="${DEFAULT_OUTPUT_DIR_BASE}"
TEMPLATE="${DEFAULT_TEMPLATE}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir-base)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Ensure the output directory for the JSONL file and the base artifact directory exist
mkdir -p "$(dirname "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_DIR_BASE}"

# Construct the command
CMD="python main_experiment.py \
    --input-file \"${INPUT_FILE}\" \
    --output-file \"${OUTPUT_FILE}\" \
    --model \"${MODEL}\" \
    --output-dir-base \"${OUTPUT_DIR_BASE}\" \
    --template \"${TEMPLATE}\""

# Display the command to be executed
echo "Executing command:"
echo "${CMD}"
echo ""

# Execute the command
eval ${CMD}

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Experiment script finished successfully."
    echo "Results saved to: ${OUTPUT_FILE}"
    echo "Task artifacts saved in subdirectories under: ${OUTPUT_DIR_BASE}"
else
    echo ""
    echo "Experiment script failed with exit code ${EXIT_CODE}."
fi

exit ${EXIT_CODE} 