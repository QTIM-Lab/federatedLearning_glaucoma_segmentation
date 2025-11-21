#!/bin/bash

# Central Training: Combined dataset training (non-federated)
# Uses train/localtraining.py
# Usage: ./centraltrain.sh [--skip-training]

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

# variables
binDriOriga_mean="0.88702821 0.5666646 0.29283205"
binDriOriga_std="0.15141807 0.17632471 0.1514714"
riga_mean="0.83512329 0.49202078 0.22558286"
riga_std="0.14766443 0.16120118 0.12717136"
messidor_mean="0.84204773 0.46466044 0.19547342"
messidor_std="0.14704964 0.15650324 0.11095415"
magrabi_mean="0.8586507  0.57209649 0.25212852"
magrabi_std="0.13919874 0.15322957 0.16213763"
binrushed_mean="0.80747745 0.51806518 0.28384791"
binrushed_std="0.14918518 0.15869727 0.12060928"
combined_mean="0.768 0.476 0.290"
combined_std="0.220 0.198 0.166"

# Training outputs go to system tmp (only created if training runs)
OUTPUT_DIRECTORY="/tmp/flglaucomaseg_train/${USER}/central"

if [ $DO_TRAIN -eq 1 ]; then
  echo "Starting training"
  # Create training output directory only when training
  mkdir -p "$OUTPUT_DIRECTORY"

  # train
  python3 ${REPO_ROOT}/engine/train/localtraining.py \
    --train_csv ${REPO_ROOT}/metadata/combined_train.csv \
    --val_csv ${REPO_ROOT}/metadata/combined_val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory $OUTPUT_DIRECTORY \
    --dataset_mean $combined_mean \
    --dataset_std $combined_std \
    --lr 0.00002 \
    --batch_size 8 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 100 \
    --patience 7 \
    --num_val_outputs_to_save 5 \
    --num_workers 16 \
    --cuda_num 1

  echo "-- train job finished --"
  sleep 5
else
  echo "[INFO] Skipping training (--skip-training flag used)."
fi

MODEL_DIR="${REPO_ROOT}/models/central"
# Handle both single checkpoint (best model only) and multiple checkpoints (full training)
CHECKPOINTS=(${MODEL_DIR}/model_epoch*_ckpt.pt)
if [ ${#CHECKPOINTS[@]} -eq 1 ]; then
    # Single checkpoint - use it directly
    LAST_CHECKPOINT="${CHECKPOINTS[0]}"
else
    # Multiple checkpoints - select the best one (highest epoch number)
    LAST_CHECKPOINT=$(ls ${MODEL_DIR}/model_epoch*_ckpt.pt 2>/dev/null | sort -V | tail -n 1)
fi

if [ ! -f "$LAST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${LAST_CHECKPOINT}"
echo "Starting testing"

TEST_OUTPUT_DIRECTORY="${REPO_ROOT}/outputs/central"
# test
python3 ${REPO_ROOT}/engine/inference.py \
    --model_path ${LAST_CHECKPOINT} \
    --input_csv ${REPO_ROOT}/metadata/combined_test.csv \
    --csv_path_col_name image_path \
    --output_root_dir $TEST_OUTPUT_DIRECTORY \
    --num_processes 1 \
    --cuda_num 1

SCORES_DIR="${REPO_ROOT}/scores/central"
mkdir -p ${SCORES_DIR}

echo "Starting evaluation for disc segmentation"
# To eval disc:
python3 ${REPO_ROOT}/engine/evaluate.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder ${REPO_ROOT}/data/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --eval_disc \
    --cuda_num 1 \
    --output_csv ${SCORES_DIR}/per_sample_disc_scores.csv \
    --model_name "central" \
    --statistical_output_dir ${REPO_ROOT}/scores

echo "Starting evaluation for cup segmentation"
# To eval cup:
python3 ${REPO_ROOT}/engine/evaluate.py \
    --prediction_folder ${TEST_OUTPUT_DIRECTORY}/outputs \
    --label_folder ${REPO_ROOT}/data/ \
    --csv_path ${TEST_OUTPUT_DIRECTORY}/results.csv \
    --cuda_num 1 \
    --output_csv ${SCORES_DIR}/per_sample_cup_scores.csv \
    --model_name "central" \
    --statistical_output_dir ${REPO_ROOT}/scores

sleep 5

echo ""
echo "-- Central training complete --"
echo "Results:"
echo "  • Models: ${REPO_ROOT}/models/central/"
echo "  • Predictions: ${TEST_OUTPUT_DIRECTORY}/"
echo "  • Scores: ${SCORES_DIR}/"
echo "  • Statistical Analysis: ${REPO_ROOT}/Statistics/"
echo "  • Plots: ${REPO_ROOT}/plots/" 