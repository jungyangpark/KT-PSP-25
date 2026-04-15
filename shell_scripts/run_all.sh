#!/bin/bash

# =============================================================================
# Master Script: Run All KT Model Experiments
#
# This script orchestrates experiments for all 10 KT models with:
# - Hyperparameter sweep (baseline)
# - Ablation study (Baseline vs +MP)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Models to run (10 models, excluding DKT2)
MODELS=(
    "dkt"
    "dkt_plus"
    "dkt_forget"
    "dkvmn"
    "skvmn"
    "sakt"
    "saint"
    "akt"
    "simplekt"
    "stablekt"
    "robustkt"
)

# Experiment mode: "sweep", "ablation", or "both"
EXP_MODE=${1:-"both"}

# Optional: Run specific model only
SPECIFIC_MODEL=${2:-""}

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

run_model() {
    local model=$1
    local mode=$2
    local log_file="${LOG_DIR}/${model}_${mode}_$(date +%Y%m%d_%H%M%S).log"

    print_header "Running $model ($mode)"

    if [ -f "${SCRIPT_DIR}/${model}.sh" ]; then
        echo "Log file: $log_file"
        bash "${SCRIPT_DIR}/${model}.sh" "$mode" 2>&1 | tee "$log_file"
        echo "Completed: $model ($mode)"
    else
        echo "WARNING: Script not found: ${SCRIPT_DIR}/${model}.sh"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

print_header "KT-PSP-25 Experiment Runner"
echo "Mode: $EXP_MODE"
echo "Models: ${MODELS[*]}"
echo "Logs: $LOG_DIR"

# If specific model is provided, run only that model
if [ -n "$SPECIFIC_MODEL" ]; then
    echo "Running specific model: $SPECIFIC_MODEL"
    run_model "$SPECIFIC_MODEL" "$EXP_MODE"
    exit 0
fi

# Run all models
START_TIME=$(date +%s)

for model in "${MODELS[@]}"; do
    run_model "$model" "$EXP_MODE"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

print_header "All Experiments Completed"
echo "Total time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
