#!/bin/bash

# Per-site Training: Individual site training (non-federated)
# Uses train/localtraining.py
#
# Usage:
#   ./persite.sh <dataset_name> [--skip-training]
#   ./persite.sh all [sequential|parallel] [--skip-training]
#   ./persite.sh --help
#
# Datasets:
#   binrushed chaksu drishti g1020 magrabi messidor origa refuge rimone

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

set -euo pipefail

show_help() {
  echo "Unified per-site runner"
  echo ""
  echo "USAGE:"
  echo "  $0 <dataset_name> [--skip-training]         # Run one dataset"
  echo "  $0 all [sequential|parallel] [--skip-training]  # Run all datasets"
  echo "  $0 --help"
  echo ""
  echo "DATASETS: binrushed chaksu drishti g1020 magrabi messidor origa refuge rimone"
  echo ""
  echo "NOTES:"
  echo "- By default, training is performed."
  echo "- Add --skip-training to skip training and use existing checkpoints."
}

datasets=("binrushed" "chaksu" "drishti" "g1020" "magrabi" "messidor" "origa" "refuge" "rimone")

# Defaults
DO_TRAIN=1
MODE="sequential"

# Parse global flags
GLOBAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --skip-training)
      DO_TRAIN=0
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      GLOBAL_ARGS+=("$arg")
      ;;
  esac
done
set -- "${GLOBAL_ARGS[@]}"

if [ $# -eq 0 ]; then
  show_help
  exit 0
fi


run_single_dataset() {
  local DATASET_NAME=$1

  echo "=== STARTING PER-SITE RUN FOR: ${DATASET_NAME} ==="
  echo "Started at: $(date)"

  # constants
  local combined_mean="0.768 0.476 0.290"
  local combined_std="0.220 0.198 0.166"

  # repo-local paths (use REPO_ROOT from top of script)
  local TRAIN_SCRIPT="${REPO_ROOT}/engine/train/localtraining.py"
  local TRAIN_CSV="${REPO_ROOT}/metadata/${DATASET_NAME}_train.csv"
  local VAL_CSV="${REPO_ROOT}/metadata/${DATASET_NAME}_val.csv"
  # Training outputs go to system tmp (only created if training runs)
  local OUTPUT_DIRECTORY="/tmp/flglaucomaseg_train/${USER}/persite/${DATASET_NAME}"

  local MODEL_DIR="${REPO_ROOT}/models/persite/${DATASET_NAME}"
  local INFER_SCRIPT="${REPO_ROOT}/engine/inference.py"
  local INPUT_CSV="${REPO_ROOT}/metadata/combined_test.csv"
  local TEST_OUTPUT_DIRECTORY="${REPO_ROOT}/outputs/persite/${DATASET_NAME}"

  local EVAL_SCRIPT="${REPO_ROOT}/engine/evaluate.py"
  local LABEL_FOLDER="${REPO_ROOT}/data"

  # Only create TEST_OUTPUT_DIRECTORY (always needed for inference)
  mkdir -p "$TEST_OUTPUT_DIRECTORY"

  if [ $DO_TRAIN -eq 1 ]; then
    echo "Starting training for dataset: $DATASET_NAME"
    # Create training output directory only when training
    mkdir -p "$OUTPUT_DIRECTORY"
    python3 "$TRAIN_SCRIPT" \
      --train_csv "$TRAIN_CSV" \
      --val_csv "$VAL_CSV" \
      --csv_img_path_col image_path \
      --csv_label_path_col label_path \
      --output_directory "$OUTPUT_DIRECTORY" \
      --dataset_mean $combined_mean \
      --dataset_std $combined_std \
      --lr 0.00002 \
      --batch_size 8 \
      --jitters 0.5 0.5 0.25 0.1 0.75 \
      --num_epochs 100 \
      --patience 7 \
      --num_val_outputs_to_save 5 \
      --num_workers 16 \
      --model_dir "$MODEL_DIR"

    echo "-- train job finished for ${DATASET_NAME} --"
    sleep 5
  else
    echo "[INFO] Skipping training (--skip-training flag used)."
  fi

  echo "Locating latest checkpoint in: ${MODEL_DIR}"
  # Wait up to 5 minutes for a checkpoint to appear if training just ran
  local timeout=300
  local start_time=$(date +%s)
  local LAST_CHECKPOINT=""
  while true; do
    if ls ${MODEL_DIR}/model_epoch*_ckpt.pt 1> /dev/null 2>&1; then
      break
    fi
    local now=$(date +%s)
    local elapsed=$((now - start_time))
    if [ $elapsed -ge $timeout ]; then
      echo "[WARN] Timeout waiting for checkpoint; proceeding with available checkpoint..."
      break
    fi
    echo "Waiting for checkpoint... (${elapsed}s)"
    sleep 10
  done

  # Handle both single checkpoint (best model only) and multiple checkpoints (full training)
  local CHECKPOINTS=(${MODEL_DIR}/model_epoch*_ckpt.pt)
  if [ ${#CHECKPOINTS[@]} -eq 1 ]; then
    # Single checkpoint - use it directly
    LAST_CHECKPOINT="${CHECKPOINTS[0]}"
  else
    # Multiple checkpoints - select the best one (highest epoch number)
    LAST_CHECKPOINT=$(ls ${MODEL_DIR}/model_epoch*_ckpt.pt 2>/dev/null | sort -V | tail -n 1 || true)
  fi

  if [ -z "$LAST_CHECKPOINT" ] || [ ! -f "$LAST_CHECKPOINT" ]; then
    echo "[ERROR] No checkpoint found in ${MODEL_DIR} (expected pattern model_epoch*_ckpt.pt)."
    return 1
  fi

  echo "Using checkpoint: ${LAST_CHECKPOINT}"
  echo "Starting testing for ${DATASET_NAME}"

  python3 "$INFER_SCRIPT" \
    --model_path "$LAST_CHECKPOINT" \
    --input_csv "$INPUT_CSV" \
    --csv_path_col_name image_path \
    --output_root_dir "$TEST_OUTPUT_DIRECTORY" \
    --num_processes 1 \
    --cuda_num 0

  sleep 5

  local SCORES_DIR="${REPO_ROOT}/scores/persite/${DATASET_NAME}"
  mkdir -p "$SCORES_DIR"

  echo "Starting PER-SAMPLE evaluation for disc segmentation"
  python3 "$EVAL_SCRIPT" \
    --prediction_folder "${TEST_OUTPUT_DIRECTORY}/outputs" \
    --label_folder "$LABEL_FOLDER" \
    --csv_path "${TEST_OUTPUT_DIRECTORY}/results.csv" \
    --eval_disc \
    --cuda_num 0 \
    --output_csv "${SCORES_DIR}/per_sample_disc_scores.csv" \
    --model_name "${DATASET_NAME}_persite" \
    --statistical_output_dir "${REPO_ROOT}/scores"

  sleep 5

  echo "Starting PER-SAMPLE evaluation for cup segmentation"
  python3 "$EVAL_SCRIPT" \
    --prediction_folder "${TEST_OUTPUT_DIRECTORY}/outputs" \
    --label_folder "$LABEL_FOLDER" \
    --csv_path "${TEST_OUTPUT_DIRECTORY}/results.csv" \
    --cuda_num 0 \
    --output_csv "${SCORES_DIR}/per_sample_cup_scores.csv" \
    --model_name "${DATASET_NAME}_persite" \
    --statistical_output_dir "${REPO_ROOT}/scores"

  echo "-- test job finished for ${DATASET_NAME} --"
  echo "=== RESULTS SUMMARY FOR ${DATASET_NAME} ==="
  echo "Training outputs: $OUTPUT_DIRECTORY"
  echo "Model checkpoints: $MODEL_DIR"
  echo "Test results: $TEST_OUTPUT_DIRECTORY"
  echo "Per-sample scores:"
  echo "  Disc: ${SCORES_DIR}/per_sample_disc_scores.csv"
  echo "  Cup:  ${SCORES_DIR}/per_sample_cup_scores.csv"
  echo "Finished at: $(date)"
  return 0
}

TARGET=$1
shift || true

if [ "$TARGET" = "all" ]; then
  MODE="${1:-sequential}"
  if [ "$MODE" != "sequential" ] && [ "$MODE" != "parallel" ]; then
    echo "[ERROR] Invalid mode '$MODE'. Use 'sequential' or 'parallel'."
    show_help
    exit 1
  fi

  echo "=== Per-site Batch Runner ==="
  echo "Mode: $MODE"
  echo "Datasets: ${datasets[*]}"
  echo "Training: $([ $DO_TRAIN -eq 1 ] && echo 'will train' || echo 'skipped (--skip-training)')"
  echo "Start: $(date)"
  echo ""

  if [ "$MODE" = "sequential" ]; then
    start_time=$(date +%s)
    failed=()
    for i in "${!datasets[@]}"; do
      ds="${datasets[$i]}"
      echo "=== DATASET $((i+1))/${#datasets[@]}: $ds ==="
      if ! run_single_dataset "$ds"; then
        failed+=("$ds")
      fi
      echo ""
      sleep 2
    done
    end_time=$(date +%s)
    dur=$((end_time - start_time))
    echo "=== SEQUENTIAL COMPLETE ==="
    echo "Total time: ${dur}s ($((dur/60))m $((dur%60))s)"
    if [ ${#failed[@]} -eq 0 ]; then
      echo "All datasets completed successfully."
    else
      echo "Failed datasets: ${failed[*]}"
    fi
  else
    echo "Running datasets in PARALLEL... (resource intensive)"
    pids=()
    start_time=$(date +%s)
    for ds in "${datasets[@]}"; do
      echo "Starting $ds in background..."
      (run_single_dataset "$ds") > "persite_${ds}.log" 2>&1 &
      pids+=("$!:$ds")
    done
    echo "All datasets started. Monitor with: tail -f persite_*.log"
    failed=()
    for entry in "${pids[@]}"; do
      pid="${entry%%:*}"
      ds="${entry##*:}"
      echo "Waiting for $ds (PID $pid)..."
      if ! wait "$pid"; then
        failed+=("$ds")
      fi
    done
    end_time=$(date +%s)
    dur=$((end_time - start_time))
    echo "=== PARALLEL COMPLETE ==="
    echo "Total time: ${dur}s ($((dur/60))m $((dur%60))s)"
    if [ ${#failed[@]} -eq 0 ]; then
      echo "All datasets completed successfully."
    else
      echo "Failed datasets: ${failed[*]}"
      echo "Check logs: persite_*.log"
    fi
  fi
else
  # Single dataset path
  name="$TARGET"
  if [[ ! " ${datasets[*]} " =~ " ${name} " ]]; then
    echo "[ERROR] Invalid dataset '$name'"
    echo "Valid datasets: ${datasets[*]}"
    show_help
    exit 1
  fi
  if run_single_dataset "$name"; then
    echo "Dataset $name completed successfully."
    exit 0
  else
    echo "Dataset $name failed."
    exit 1
  fi
fi

sleep 5

echo ""
echo "-- Per-site training complete --"
echo "Results:"
echo "  • Statistical Analysis: ${REPO_ROOT}/Statistics/"
echo "  • Plots: ${REPO_ROOT}/plots/"
