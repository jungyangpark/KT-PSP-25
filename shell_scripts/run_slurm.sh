#!/bin/bash

# =============================================================================
# SLURM Job Submission Script for KT Experiments
# Submit individual model experiments as separate SLURM jobs
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs/slurm"

# Create log directory
mkdir -p "$LOG_DIR"

# SLURM Configuration (modify as needed)
PARTITION=${PARTITION:-"base_suma_rtx3090"}
QOS=${QOS:-"base_qos"}
GRES=${GRES:-"gpu:1"}
TIME=${TIME:-"48:00:00"}
MEM=${MEM:-"32G"}

# Models to run
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

# Experiment mode
EXP_MODE=${1:-"both"}

# =============================================================================
# Functions
# =============================================================================

submit_job() {
    local model=$1
    local mode=$2
    local job_name="${model}_${mode}"
    local output_file="${LOG_DIR}/${job_name}_%j.out"
    local error_file="${LOG_DIR}/${job_name}_%j.err"

    echo "Submitting: $job_name"

    sbatch \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --qos="$QOS" \
        --gres="$GRES" \
        --time="$TIME" \
        --mem="$MEM" \
        --output="$output_file" \
        --error="$error_file" \
        --wrap="cd ${SCRIPT_DIR}/../pykt-toolkit/examples && bash ${SCRIPT_DIR}/${model}.sh $mode"

    echo "  -> Output: $output_file"
}

# =============================================================================
# Main Execution
# =============================================================================

echo "============================================================"
echo "KT-PSP-25 SLURM Job Submission"
echo "============================================================"
echo ""
echo "Mode: $EXP_MODE"
echo "Partition: $PARTITION"
echo "QOS: $QOS"
echo "GPU: $GRES"
echo "Time: $TIME"
echo "Memory: $MEM"
echo ""

# Check if specific model is provided
if [ -n "$2" ] && [ "$2" != "all" ]; then
    submit_job "$2" "$EXP_MODE"
    exit 0
fi

# Submit all models
for model in "${MODELS[@]}"; do
    submit_job "$model" "$EXP_MODE"
done

echo ""
echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."
