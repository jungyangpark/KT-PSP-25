#!/bin/bash

# =============================================================================
# SimpleKT Model Training Script
# Supports: Baseline, +MP ablation study and hyperparameter sweep
# =============================================================================

# Configuration
DATASET_NAME="custom"
MODEL_NAME="simplekt"
USE_WANDB=1
SEED=42
COLD_START_LEN=0
SAVE_DIR="saved_model"

# SimpleKT specific hyperparameters
D_MODEL=256
D_FF=256
NUM_ATTN_HEADS=4
N_BLOCKS=2
FINAL_FC_DIM=256
FINAL_FC_DIM2=256
NUM_LAYERS=2
NHEADS=4
LOSS1=0.5
LOSS2=0.5
LOSS3=0.5
START=50

# Experiment mode
EXP_MODE=${1:-"both"}

# Ablation configurations
MP_CONFIGS_ABLATION=(0 8)

# Hyperparameter sweep configurations
LEARNING_RATES=(1e-3 5e-3 1e-4 5e-4)
DROPOUTS=(0.05 0.1 0.3 0.5)
FOLDS=(0 1 2 3 4)

# Best hyperparameters for ablation
BEST_LR=${2:-"1e-4"}
BEST_DROPOUT=${3:-"0.1"}

# =============================================================================
# Functions
# =============================================================================

run_single_experiment() {
    local lr=$1
    local dropout=$2
    local fold=$3
    local use_mp=$4

    echo "  [SimpleKT] fold=$fold, lr=$lr, dropout=$dropout, use_mp=$use_mp"

    python wandb_simplekt_train.py \
        --dataset_name $DATASET_NAME \
        --model_name $MODEL_NAME \
        --use_wandb $USE_WANDB \
        --seed $SEED \
        --cold_start_len $COLD_START_LEN \
        --save_dir $SAVE_DIR \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --num_attn_heads $NUM_ATTN_HEADS \
        --n_blocks $N_BLOCKS \
        --final_fc_dim $FINAL_FC_DIM \
        --final_fc_dim2 $FINAL_FC_DIM2 \
        --num_layers $NUM_LAYERS \
        --nheads $NHEADS \
        --loss1 $LOSS1 \
        --loss2 $LOSS2 \
        --loss3 $LOSS3 \
        --start $START \
        --learning_rate $lr \
        --fold $fold \
        --dropout $dropout \
        --use_mp $use_mp \
        --archi 1
}

run_ablation_study() {
    echo "=========================================="
    echo "SimpleKT Ablation Study"
    echo "Learning Rate: $BEST_LR, Dropout: $BEST_DROPOUT"
    echo "=========================================="

    for use_mp in "${MP_CONFIGS_ABLATION[@]}"; do
        case $use_mp in
            0) exp_name="Baseline" ;;
            8) exp_name="+MP" ;;
        esac
        echo "--- Running: $exp_name (use_mp=$use_mp) ---"

        for fold in "${FOLDS[@]}"; do
            run_single_experiment $BEST_LR $BEST_DROPOUT $fold $use_mp
        done

        rm -rf ${SAVE_DIR}/${DATASET_NAME}_${MODEL_NAME}_qid_saved_model_${SEED}_*
    done
}

run_hyperparameter_sweep() {
    echo "=========================================="
    echo "SimpleKT Hyperparameter Sweep"
    echo "=========================================="

    local use_mp=0

    for dropout in "${DROPOUTS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            echo "--- Sweep: lr=$lr, dropout=$dropout ---"
            for fold in "${FOLDS[@]}"; do
                run_single_experiment $lr $dropout $fold $use_mp
            done
        done
        rm -rf ${SAVE_DIR}/${DATASET_NAME}_${MODEL_NAME}_qid_saved_model_${SEED}_*
    done
}

# =============================================================================
# Main Execution
# =============================================================================

cd "$(dirname "$0")/../pykt-toolkit/examples"

echo "Starting SimpleKT experiments..."
echo "Mode: $EXP_MODE"

case $EXP_MODE in
    "ablation")
        run_ablation_study
        ;;
    "sweep")
        run_hyperparameter_sweep
        ;;
    "both")
        run_hyperparameter_sweep
        run_ablation_study
        ;;
    *)
        echo "Unknown mode: $EXP_MODE"
        echo "Usage: $0 [ablation|sweep|both] [best_lr] [best_dropout]"
        exit 1
        ;;
esac

echo "SimpleKT experiments completed!"
