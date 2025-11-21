#!/bin/bash

# Pipeline 3: Democratic/Unweighted FedAvg Training (all datasets equal)
# Uses train/pipeline3.py
# Usage: ./pipeline3.sh [--skip-training]

# Activate virtual environment if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment: ${REPO_ROOT}/.venv"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using existing virtual environment: ${VIRTUAL_ENV}"
else
    echo "Warning: No virtual environment found. Install dependencies with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# Parse flags
DO_TRAIN=1
for arg in "$@"; do
  case "$arg" in
    --skip-training)
      DO_TRAIN=0
      ;;
    --help|-h)
      echo "Usage: $0 [--skip-training]"
      echo "  --skip-training: Skip training and use existing model"
      exit 0
      ;;
  esac
done

# Democratic/Unweighted FedAvg - all datasets get equal weighting

# variables
combined_mean="0.768 0.476 0.290"
combined_std="0.220 0.198 0.166"

# Training outputs go to system tmp (only created if training runs)
OUTPUT_DIRECTORY="/tmp/flglaucomaseg_train/${USER}/pipeline3"

if [ $DO_TRAIN -eq 1 ]; then
  echo "Starting Pipeline 3: Democratic/Unweighted FedAvg training"
  # Create training output directory only when training
  mkdir -p "$OUTPUT_DIRECTORY"

  # train
  python3 ${REPO_ROOT}/engine/train/pipeline3.py \
    --train_csv ${REPO_ROOT}/metadata/binrushed_train.csv \
                ${REPO_ROOT}/metadata/chaksu_train.csv \
                ${REPO_ROOT}/metadata/drishti_train.csv \
                ${REPO_ROOT}/metadata/g1020_train.csv \
                ${REPO_ROOT}/metadata/magrabi_train.csv \
                ${REPO_ROOT}/metadata/messidor_train.csv \
                ${REPO_ROOT}/metadata/origa_train.csv \
                ${REPO_ROOT}/metadata/refuge_train.csv \
                ${REPO_ROOT}/metadata/rimone_train.csv \
    --val_csv ${REPO_ROOT}/metadata/binrushed_val.csv \
                ${REPO_ROOT}/metadata/chaksu_val.csv \
                ${REPO_ROOT}/metadata/drishti_val.csv \
                ${REPO_ROOT}/metadata/g1020_val.csv \
                ${REPO_ROOT}/metadata/magrabi_val.csv \
                ${REPO_ROOT}/metadata/messidor_val.csv \
                ${REPO_ROOT}/metadata/origa_val.csv \
                ${REPO_ROOT}/metadata/refuge_val.csv \
                ${REPO_ROOT}/metadata/rimone_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory $OUTPUT_DIRECTORY \
    --dataset_mean $combined_mean \
    --dataset_std $combined_std \
    --lr 0.00002 \
    --batch_size 8 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --fl_rounds 10 \
    --patience 7 \
    --num_val_outputs_to_save 5 \
    --num_workers 16 \
    --local_sites_training_epochs 20 \
    --fl_patience 3 

  echo "-- train job finished --"
else
  echo "[INFO] Skipping training (--skip-training flag used)."
fi

MODEL_DIR="${REPO_ROOT}/models/pipeline3"
# Handle both single checkpoint (best model only) and multiple checkpoints (full training)
CHECKPOINTS=(${MODEL_DIR}/best_global_model_round_*.pth)
if [ ${#CHECKPOINTS[@]} -eq 1 ]; then
    # Single checkpoint - use it directly
    LAST_CHECKPOINT="${CHECKPOINTS[0]}"
else
    # Multiple checkpoints - select the best one (highest round number with lowest metric)
    LAST_CHECKPOINT=$(ls ${MODEL_DIR}/best_global_model_round_*.pth 2>/dev/null | sort -V | tail -n 1)
fi

if [ ! -f "$LAST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${LAST_CHECKPOINT}"
echo "Starting testing"

TEST_OUTPUT_DIRECTORY="${REPO_ROOT}/outputs/pipeline3"
# test
python3 ${REPO_ROOT}/engine/inference.py \
    --model_path ${LAST_CHECKPOINT} \
    --input_csv ${REPO_ROOT}/metadata/combined_test.csv \
    --csv_path_col_name image_path \
    --output_root_dir $TEST_OUTPUT_DIRECTORY \
    --num_processes 1 \
    --cuda_num 0

SCORES_DIR="${REPO_ROOT}/scores/pipeline3"
mkdir -p ${SCORES_DIR}

echo "Starting evaluation for disc segmentation"
# To eval disc:
python3 ${REPO_ROOT}/engine/evaluate.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder ${REPO_ROOT}/data/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --eval_disc \
    --cuda_num 0 \
    --output_csv ${SCORES_DIR}/per_sample_disc_scores.csv \
    --model_name "pipeline3" \
    --statistical_output_dir ${REPO_ROOT}/scores

echo "Starting evaluation for cup segmentation"
# To eval cup:
python3 ${REPO_ROOT}/engine/evaluate.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder ${REPO_ROOT}/data/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --cuda_num 0 \
    --output_csv ${SCORES_DIR}/per_sample_cup_scores.csv \
    --model_name "pipeline3" \
    --statistical_output_dir ${REPO_ROOT}/scores

sleep 5

echo ""
echo "-- Pipeline 3 complete --"
echo "Results:"
echo "  • Models: ${REPO_ROOT}/models/pipeline3/"
echo "  • Predictions: ${TEST_OUTPUT_DIRECTORY}/"
echo "  • Scores: ${SCORES_DIR}/"
echo "  • Statistical Analysis: ${REPO_ROOT}/Statistics/"
echo "  • Plots: ${REPO_ROOT}/plots/" 