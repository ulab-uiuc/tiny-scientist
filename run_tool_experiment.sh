#!/bin/bash

# Set default values
MODEL="gpt-4o"
DOMAIN="medical"
OUTPUT_DIR="./output/tool_experiments"
MAX_ITER=5
SAFE_ONLY=false
USE_SAFE_INSTRUCTOR=false
USE_MALICIOUS_INSTRUCTOR=false

# Help function
show_help() {
    echo "Usage: run_tool_experiment.sh [options]"
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -m, --model MODEL       Specify the LLM model (default: gpt-4o)"
    echo "  -d, --domain DOMAIN     Set domain: medical, physics (default: medical)"
    echo "  -o, --output DIR        Set output directory (default: ./output/tool_experiments)"
    echo "  --max-iter NUM          Set maximum iterations (default: 5)"
    echo "  --safe-only             Only run experiments that pass safety checks"
    echo "  --safe-instructor       Enable safe experiment instructor to guide parameter generation"
    echo "  --malicious-instructor  Enable malicious experiment instructor to attempt safety bypass"
    echo
    echo "Predefined experiments:"
    echo "  --medical               Run medical tool experiments"
    echo "  --physics               Run physics tool experiments"
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
        --safe-only)
            SAFE_ONLY=true
            shift
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

# Construct command
CMD="python tool_experimenter.py --model \"$MODEL\" --domain $DOMAIN --output-dir \"$OUTPUT_DIR\" --max-iterations $MAX_ITER"

# Add safe-only flag if specified
if [ "$SAFE_ONLY" = true ]; then
    CMD="$CMD --safe-only"
fi

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