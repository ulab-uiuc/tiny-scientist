#!/bin/bash

# Set default values
MODEL="gpt-4o"
DOMAIN="general"
OUTPUT_DIR="./output"
MAX_ITER=10
TEMPLATE="acl"
INTENT=""

# Help function
show_help() {
    echo "Usage: run_experiment.sh [options]"
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -m, --model MODEL       Specify the LLM model (default: gpt-4o)"
    echo "  -d, --domain DOMAIN     Set domain: chemistry, physics, general (default: general)"
    echo "  -i, --intent TEXT       Specify experiment intent"
    echo "  -o, --output DIR        Set output directory (default: ./output)"
    echo "  --max-iter NUM          Set maximum iterations (default: 10)"
    echo "  -t, --template FORMAT   Paper template: acl, iclr (default: acl)"
    echo
    echo "Predefined experiment examples:"
    echo "  --chemistry             Run chemistry experiment (solubility study)"
    echo "  --physics               Run physics experiment (thermal-electrical property comparison)"
    echo "  --general               Run general ML experiment"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -i|--intent)
            INTENT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        -t|--template)
            TEMPLATE="$2"
            shift 2
            ;;
        --chemistry)
            DOMAIN="chemistry"
            INTENT="Investigate NaCl solubility in water and ethanol at different temperatures"
            shift
            ;;
        --physics)
            DOMAIN="physics"
            INTENT="Compare the relationship between thermal conductivity and electrical resistivity of different materials"
            shift
            ;;
        --general)
            DOMAIN="general"
            INTENT="Explore the performance of simple machine learning models on small datasets"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Construct command
CMD="python example_react_experiment.py --model \"$MODEL\" --domain $DOMAIN --output-dir \"$OUTPUT_DIR\" --max-iterations $MAX_ITER --template $TEMPLATE"

# Add intent if specified
if [ -n "$INTENT" ]; then
    CMD="$CMD --intent \"$INTENT\""
fi

# Display command to be executed
echo "Executing: $CMD"
echo "Starting experiment..."

# Execute the command
eval $CMD

# Exit with the same status as the Python script
exit $? 