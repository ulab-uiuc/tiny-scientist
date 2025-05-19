#!/bin/bash

# Set default values
MODEL="gpt-4o"
DOMAIN="is"
OUTPUT_DIR="./output/tool_experiments_is"
MAX_ITER=5
USE_SAFE_INSTRUCTOR=false
USE_MALICIOUS_INSTRUCTOR=false

# Help function
show_help() {
    echo "Usage: run_tool_experiment.sh [options]"
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -m, --model MODEL       Specify the LLM model (default: gpt-4o)"
    echo "  -d, --domain DOMAIN     Set domain: medical, physics, biology, chemical, material, is (default: material)"
    echo "  -o, --output DIR        Set output directory (default: ./output/tool_experiments)"
    echo "  --max-iter NUM          Set maximum iterations (default: 5)"
    echo "  --safe-instructor       Enable safe experiment instructor to guide parameter generation"
    echo "  --malicious-instructor  Enable malicious experiment instructor to attempt safety bypass"
    echo
    echo "Predefined experiments:"
    echo "  --medical               Run medical tool experiments"
    echo "  --physics               Run physics tool experiments"
    echo "  --biology               Run biology tool experiments"
    echo "  --chemical              Run chemical tool experiments"
    echo "  --material              Run material tool experiments"
    echo "  --is                    Run information system tool experiments"
    echo
    echo "Instructor combinations:"
    echo "  --secure-test           Enable safe instructor only (conservative parameters)"
    echo "  --penetration-test      Enable malicious instructor only (attempting to bypass safety checks)"
    echo "  --full-benchmark        Enable both instructors for comprehensive testing"
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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --safe-instructor)
            USE_SAFE_INSTRUCTOR=true
            shift
            ;;
        --malicious-instructor)
            USE_MALICIOUS_INSTRUCTOR=true
            shift
            ;;
        --medical)
            DOMAIN="medical"
            shift
            ;;
        --physics)
            DOMAIN="physics"
            shift
            ;;
        --biology)
            DOMAIN="biology"
            shift
            ;;
        --material)
            DOMAIN="material"
            shift
            ;;
        --chemical)
            DOMAIN="chemical"
            shift
            ;;
        --is)
            DOMAIN="is"
            shift
            ;;
        --secure-test)
            USE_SAFE_INSTRUCTOR=true
            USE_MALICIOUS_INSTRUCTOR=false
            shift
            ;;
        --penetration-test)
            USE_SAFE_INSTRUCTOR=false
            USE_MALICIOUS_INSTRUCTOR=true
            shift
            ;;
        --full-benchmark)
            USE_SAFE_INSTRUCTOR=true
            USE_MALICIOUS_INSTRUCTOR=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set output directory based on instructor flags
if [[ "$USE_SAFE_INSTRUCTOR" == true && "$USE_MALICIOUS_INSTRUCTOR" == false ]]; then
  OUTPUT_DIR="./output/tool_experiments/safe"
elif [[ "$USE_SAFE_INSTRUCTOR" == false && "$USE_MALICIOUS_INSTRUCTOR" == true ]]; then
  OUTPUT_DIR="./output/tool_experiments/malicious"
elif [[ "$USE_SAFE_INSTRUCTOR" == true && "$USE_MALICIOUS_INSTRUCTOR" == true ]]; then
  OUTPUT_DIR="./output/tool_experiments/both"
else
  OUTPUT_DIR="./output/tool_experiments/origin"
fi

# Construct command
CMD="python tool_experimenter.py --model \"$MODEL\" --domain $DOMAIN --output-dir \"$OUTPUT_DIR\" --max-iterations $MAX_ITER"

# Add instructor flags if specified
if [ "$USE_SAFE_INSTRUCTOR" = true ]; then
    CMD="$CMD --use-safe-instructor"
fi

if [ "$USE_MALICIOUS_INSTRUCTOR" = true ]; then
    CMD="$CMD --use-malicious-instructor"
fi

# Display command to be executed
echo "Executing: $CMD"
echo "Starting tool experiment..."
echo "Instructors: Safe=${USE_SAFE_INSTRUCTOR}, Malicious=${USE_MALICIOUS_INSTRUCTOR}"

# Execute the command
eval $CMD

# Exit with the same status as the Python script
exit $? 