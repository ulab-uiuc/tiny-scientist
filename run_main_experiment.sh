#!/bin/bash

# Default values for general settings
DEFAULT_MODEL="gpt-3.5-turbo"
DEFAULT_OUTPUT_DIR_BASE="./output/main_experiments_run"
DEFAULT_TEMPLATE="acl"
# DEFAULT_ENABLE_MALICIOUS_AGENTS and DEFAULT_ENABLE_DEFENSE_AGENT will be handled by flags

# --- Configuration for batch processing ---
BASE_DATA_DIR="./data/ScienceSafetyData/Dataset"
INPUT_FILES_BASENAMES=("bio.json" "chem.json" "is.json" "phy.json" "med.json" "material.json")
CORRESPONDING_DOMAINS=("biology" "chemistry" "information_science" "physics" "medicine" "materials")
# Suffixes for output files, can be same as domain or more specific
OUTPUT_NAME_SUFFIXES=("bio" "chem" "is" "phy" "med" "materials") 
#VALID_DOMAINS=("physics" "medicine" "materials" "information_science" "chemistry" "biology")
# --- End of Configuration for batch processing ---

# Global flags, can be overridden by command line arguments if implemented
# For now, these are effectively constants for all runs in the batch unless you add CLI parsing for them
ENABLE_MALICIOUS_AGENTS_FLAG=false # Set to true or false as needed for the batch
ENABLE_DEFENSE_AGENT_FLAG=true    # Set to true or false as needed for the batch

# Help function
show_help() {
    echo "Usage: run_main_experiment.sh [options]"
    echo ""
    echo "This script runs main_experiment.py in batch mode for predefined datasets and domains."
    echo "Input files, domains, and output file names are configured internally within the script."
    echo ""
    echo "Options:"
    echo "  -h, --help                   Show this help message and exit."
    echo "  --model MODEL_NAME           LLM model to use. (Default: ${DEFAULT_MODEL})"
    echo "  --output-dir-base DIR_PATH   Base directory for task artifacts. (Default: ${DEFAULT_OUTPUT_DIR_BASE})"
    echo "  --template TEMPLATE_NAME     Paper template. (Default: ${DEFAULT_TEMPLATE})"
    echo "  --enable-malicious-agents    Pass this flag to enable malicious agents for all runs."
    echo "  --enable-defense-agent       Pass this flag to enable the defense agent for all runs."
    echo ""
    echo "Example (uses internal batch configuration):"
    echo "  ./run_main_experiment.sh --model gpt-4o --enable-defense-agent"
}

# Initialize variables with default values for script-level options
MODEL="${DEFAULT_MODEL}"
OUTPUT_DIR_BASE="${DEFAULT_OUTPUT_DIR_BASE}"
TEMPLATE="${DEFAULT_TEMPLATE}"

# Initialize boolean flags from global settings
MALICIOUS_ARG=""
DEFENSE_ARG=""

if [ "$ENABLE_MALICIOUS_AGENTS_FLAG" = true ] ; then
    MALICIOUS_ARG="--enable-malicious-agents"
fi
if [ "$ENABLE_DEFENSE_AGENT_FLAG" = true ] ; then
    DEFENSE_ARG="--enable-defense-agent"
fi

# Parse command-line arguments for global settings
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
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
        --enable-malicious-agents)
            MALICIOUS_ARG="--enable-malicious-agents"
            shift # Consume flag
            ;;
        --enable-defense-agent)
            DEFENSE_ARG="--enable-defense-agent"
            shift # Consume flag
            ;;
        *) # Unknown option
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Ensure the base output directory for JSONL files and task artifacts exist
MAIN_OUTPUT_DIR="./output" # General output directory for JSONL files
mkdir -p "${MAIN_OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR_BASE}"

# Generate a model name suffix for output files (e.g., gpt_3_5_turbo)
MODEL_NAME_SUFFIX=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g' | sed 's/_*$//g')

# Loop through the configured files and domains
for i in "${!INPUT_FILES_BASENAMES[@]}"; do
    CURRENT_INPUT_BASENAME="${INPUT_FILES_BASENAMES[i]}"
    CURRENT_DOMAIN="${CORRESPONDING_DOMAINS[i]}"
    CURRENT_OUTPUT_SUFFIX="${OUTPUT_NAME_SUFFIXES[i]}"

    INPUT_FILE_PATH="${BASE_DATA_DIR}/${CURRENT_INPUT_BASENAME}"
    
    # Construct output file name
    DEFENSE_STATUS_SUFFIX=""
    if [ -n "${DEFENSE_ARG}" ]; then
        DEFENSE_STATUS_SUFFIX="_defense_on"
    else
        DEFENSE_STATUS_SUFFIX="_defense_off"
    fi
    MALICIOUS_STATUS_SUFFIX=""
    if [ -n "${MALICIOUS_ARG}" ]; then
        MALICIOUS_STATUS_SUFFIX="_malicious_on"
    fi

    OUTPUT_JSONL_FILE="${MAIN_OUTPUT_DIR}/results_${CURRENT_OUTPUT_SUFFIX}_${MODEL_NAME_SUFFIX}${DEFENSE_STATUS_SUFFIX}${MALICIOUS_STATUS_SUFFIX}.jsonl"

    echo "--------------------------------------------------------------------"
    echo "Starting experiment for: ${CURRENT_INPUT_BASENAME} (Domain: ${CURRENT_DOMAIN})"
    echo "Input file: ${INPUT_FILE_PATH}"
    echo "Output JSONL: ${OUTPUT_JSONL_FILE}"
    echo "Model: ${MODEL}"
    echo "Malicious Agents: $(if [ -n "${MALICIOUS_ARG}" ]; then echo "Enabled"; else echo "Disabled"; fi)"
    echo "Defense Agent: $(if [ -n "${DEFENSE_ARG}" ]; then echo "Enabled"; else echo "Disabled"; fi)"
    echo "--------------------------------------------------------------------"

    # Construct the command for main_experiment.py
    CMD="python main_experiment.py \
        --input-file \"${INPUT_FILE_PATH}\" \
        --output-file \"${OUTPUT_JSONL_FILE}\" \
        --model \"${MODEL}\" \
        --output-dir-base \"${OUTPUT_DIR_BASE}/${CURRENT_OUTPUT_SUFFIX}_run\" \
        --template \"${TEMPLATE}\" \
        --domain \"${CURRENT_DOMAIN}\" \
        ${MALICIOUS_ARG} \
        ${DEFENSE_ARG}"

    # Display the command to be executed
    echo "Executing command:"
    echo "${CMD}"
    echo ""

    # Execute the command
    eval ${CMD}

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "Experiment for ${CURRENT_INPUT_BASENAME} finished successfully."
        echo "Results saved to: ${OUTPUT_JSONL_FILE}"
    else
        echo ""
        echo "Experiment for ${CURRENT_INPUT_BASENAME} failed with exit code ${EXIT_CODE}."
    fi
    echo "--------------------------------------------------------------------"
    echo ""

done

echo "All configured experiments have been processed."
exit 0 