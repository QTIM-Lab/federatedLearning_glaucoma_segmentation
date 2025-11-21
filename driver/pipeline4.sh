#!/bin/bash

# Pipeline 4: FL Fine-tuning Script (single-round federated averaging)
# Uses train/pipeline4.py
# 
# Usage:
#   ./pipeline4.sh <dataset_name> [--skip-training]           # Run single dataset
#   ./pipeline4.sh all [sequential] [--skip-training]         # Run all datasets sequentially (default)
#   ./pipeline4.sh all parallel [--skip-training]             # Run all datasets in parallel
#   ./pipeline4.sh --help                                     # Show help

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

# Parse global flags first
DO_TRAIN=1
ARGS_CLEANED=()
for arg in "$@"; do
  case "$arg" in
    --skip-training)
      DO_TRAIN=0
      ;;
    *)
      ARGS_CLEANED+=("$arg")
      ;;
  esac
done
set -- "${ARGS_CLEANED[@]}"

# Function to display help
show_help() {
    echo "Pipeline4: Unified FL Fine-tuning Script"
    echo ""
    echo "USAGE:"
    echo "  $0 <dataset_name> [--skip-training]           # Run single dataset"
    echo "  $0 all [sequential] [--skip-training]         # Run all datasets sequentially (default)"
    echo "  $0 all parallel [--skip-training]             # Run all datasets in parallel"
    echo "  $0 --help                                     # Show this help"
    echo ""
    echo "AVAILABLE DATASETS:"
    echo "  binrushed, chaksu, drishti, g1020, magrabi, messidor, origa, refuge, rimone"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 binrushed                # Run binrushed dataset only"
    echo "  $0 all                      # Run all datasets sequentially"
    echo "  $0 all parallel             # Run all datasets in parallel (resource intensive)"
    echo "  $0 all --skip-training      # Run all without training (use existing models)"
}

# Resolve repository-relative paths (everything contained in repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to run a single dataset
run_single_dataset() {
    local DATASET_NAME=$1
    
    echo "=== STARTING FL FINE-TUNING FOR: $DATASET_NAME ==="
    echo "Started at: $(date)"
    

    # variables
    combined_mean="0.768 0.476 0.290"
    combined_std="0.220 0.198 0.166"

    # Training outputs go to system tmp (only created if training runs)
    OUTPUT_DIRECTORY="/tmp/flglaucomaseg_train/${USER}/pipeline4/${DATASET_NAME}"
    MODEL_DIR="${REPO_ROOT}/models/pipeline4/${DATASET_NAME}"
    TEST_OUTPUT_DIRECTORY="${REPO_ROOT}/outputs/pipeline4/${DATASET_NAME}"
    SCORES_DIR="${REPO_ROOT}/scores/pipeline4/${DATASET_NAME}"

    # Create necessary directories (except OUTPUT_DIRECTORY which is only for training)
    mkdir -p $MODEL_DIR
    mkdir -p $TEST_OUTPUT_DIRECTORY
    mkdir -p $SCORES_DIR

    if [ $DO_TRAIN -eq 1 ]; then
      echo "Starting training for dataset: $DATASET_NAME"
      # Create training output directory only when training
      mkdir -p $OUTPUT_DIRECTORY

      # choose pretrained global model from local models directory if available
      PRETRAINED_GLOBAL=$(ls "${REPO_ROOT}/models/pipeline3"/best_global_model_round_*.pth 2>/dev/null | sort -V | tail -n 1)
      if [ -z "$PRETRAINED_GLOBAL" ]; then
          PRETRAINED_GLOBAL=$(ls "${REPO_ROOT}/models"/*/best_global_model_round_*.pth 2>/dev/null | sort -V | tail -n 1)
      fi

      # train
      python3 "${REPO_ROOT}/engine/train/pipeline4.py" \
        --train_csv "${REPO_ROOT}/metadata/${DATASET_NAME}_train.csv" \
        --val_csv "${REPO_ROOT}/metadata/${DATASET_NAME}_val.csv" \
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
        --model_dir "$MODEL_DIR" \
        ${PRETRAINED_GLOBAL:+--pretrained_global_model_path "$PRETRAINED_GLOBAL"} \
        --cuda_num 1

      echo "-- train job finished for $DATASET_NAME --"
      sleep 5
    else
      echo "[INFO] Skipping training for $DATASET_NAME (--skip-training flag used)."
    fi

    # Wait for checkpoint to be created with a timeout (only if training just ran)
    echo "Waiting for checkpoint to be created..."
    timeout=300  # 5 minutes timeout
    start_time=$(date +%s)
    while true; do
        # Check for any best_global_model_epoch*.pth file in the model directory
        if ls ${MODEL_DIR}/best_global_model_epoch*.pth 1> /dev/null 2>&1; then
            echo "Checkpoint found"
            break
        fi
        
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            echo "Timeout waiting for checkpoint. Proceeding with available checkpoint..."
            break
        fi
        
        echo "Still waiting for checkpoint... ($elapsed seconds elapsed)"
        sleep 10
    done

    # Handle both single checkpoint (best model only) and multiple checkpoints (full training)
    CHECKPOINTS=(${MODEL_DIR}/best_global_model_epoch*.pth)
    if [ ${#CHECKPOINTS[@]} -eq 1 ]; then
        # Single checkpoint - use it directly
        LAST_CHECKPOINT="${CHECKPOINTS[0]}"
    else
        # Multiple checkpoints - select the best one (highest epoch number)
        LAST_CHECKPOINT=$(ls ${MODEL_DIR}/best_global_model_epoch*.pth 2>/dev/null | sort -V | tail -n 1)
    fi

    if [ ! -f "$LAST_CHECKPOINT" ] || [ "$LAST_CHECKPOINT" = "${MODEL_DIR}/best_global_model_epoch*.pth" ]; then
        echo "Error: No checkpoint found in ${MODEL_DIR}"
        return 1
    fi

    echo "Using checkpoint: ${LAST_CHECKPOINT}"
    echo "Starting testing for $DATASET_NAME"

    # test
    python3 "${REPO_ROOT}/engine/inference.py" \
        --model_path "${LAST_CHECKPOINT}" \
        --input_csv "${REPO_ROOT}/metadata/combined_test.csv" \
        --csv_path_col_name image_path \
        --output_root_dir "$TEST_OUTPUT_DIRECTORY" \
        --num_processes 1 \
        --cuda_num 1

    sleep 5

    echo "Starting PER-SAMPLE evaluation for disc segmentation"
    # Per-sample eval for disc:
    python3 "${REPO_ROOT}/engine/evaluate.py" \
        --prediction_folder "${TEST_OUTPUT_DIRECTORY}/outputs" \
        --label_folder "${REPO_ROOT}/data/" \
        --csv_path "${TEST_OUTPUT_DIRECTORY}/results.csv" \
        --eval_disc \
        --cuda_num 1 \
        --output_csv "${SCORES_DIR}/per_sample_disc_scores.csv" \
        --model_name "${DATASET_NAME}_fl_finetuned" \
        --statistical_output_dir "${REPO_ROOT}/scores"

    echo "Starting PER-SAMPLE evaluation for cup segmentation"
    # Per-sample eval for cup:
    python3 "${REPO_ROOT}/engine/evaluate.py" \
        --prediction_folder "${TEST_OUTPUT_DIRECTORY}/outputs" \
        --label_folder "${REPO_ROOT}/data/" \
        --csv_path "${TEST_OUTPUT_DIRECTORY}/results.csv" \
        --cuda_num 1 \
        --output_csv "${SCORES_DIR}/per_sample_cup_scores.csv" \
        --model_name "${DATASET_NAME}_fl_finetuned" \
        --statistical_output_dir "${REPO_ROOT}/scores"

    echo "-- test job finished for $DATASET_NAME --"
    echo ""
    echo "=== RESULTS SUMMARY FOR $DATASET_NAME ==="
    echo "Training outputs: $OUTPUT_DIRECTORY"
    echo "Model checkpoints: $MODEL_DIR"
    echo "Test results: $TEST_OUTPUT_DIRECTORY"
    echo "Per-sample scores:"
    echo "  Disc: ${SCORES_DIR}/per_sample_disc_scores.csv"
    echo "  Cup:  ${SCORES_DIR}/per_sample_cup_scores.csv"
    echo "Finished at: $(date)"
    
    return 0
}

# Main script logic
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Array of all datasets
datasets=("binrushed" "chaksu" "drishti" "g1020" "magrabi" "messidor" "origa" "refuge" "rimone")

# Check if running all datasets
if [ "$1" = "all" ]; then
    MODE=${2:-sequential}  # Default to sequential if no second argument
    
    if [[ "$MODE" != "sequential" && "$MODE" != "parallel" ]]; then
        echo "Error: Invalid mode '$MODE'. Use 'sequential' or 'parallel'."
        show_help
        exit 1
    fi
    
    echo "=== Pipeline4 Batch Runner ==="
    echo "Mode: $MODE"
    echo "Datasets: ${datasets[*]}"
    echo "Total datasets: ${#datasets[@]}"
    echo ""
    
    if [ "$MODE" = "sequential" ]; then
        echo "Running datasets SEQUENTIALLY..."
        echo ""
        
        start_time=$(date +%s)
        failed_datasets=()
        
        for i in "${!datasets[@]}"; do
            dataset="${datasets[$i]}"
            echo "=== DATASET $((i+1))/${#datasets[@]}: $dataset ==="
            
            dataset_start=$(date +%s)
            
            # Run the dataset
            run_single_dataset "$dataset"
            exit_code=$?
            
            dataset_end=$(date +%s)
            dataset_duration=$((dataset_end - dataset_start))
            
            if [ $exit_code -eq 0 ]; then
                echo "$dataset completed successfully in ${dataset_duration}s"
            else
                echo "$dataset failed with exit code $exit_code"
                failed_datasets+=("$dataset")
            fi
            
            echo ""
            # Small delay between datasets
            sleep 2
        done
        
        end_time=$(date +%s)
        total_duration=$((end_time - start_time))
        
        echo "=== SEQUENTIAL EXECUTION COMPLETE ==="
        echo "Total time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
        
        if [ ${#failed_datasets[@]} -eq 0 ]; then
            echo "All datasets completed successfully"
        else
            echo "Failed datasets: ${failed_datasets[*]}"
        fi
        
    elif [ "$MODE" = "parallel" ]; then
        echo "Running datasets in PARALLEL..."
        echo "WARNING: This will use significant computational resources"
        echo ""
        
        # Array to store background process PIDs
        pids=()
        
        start_time=$(date +%s)
        
        for dataset in "${datasets[@]}"; do
            echo "Starting $dataset in background..."
            
            # Run each dataset in background and capture PID
            (run_single_dataset "$dataset") > "pipeline4_${dataset}.log" 2>&1 &
            pid=$!
            pids+=($pid)
            
            echo "  $dataset started with PID $pid"
        done
        
        echo ""
        echo "All datasets started. Waiting for completion..."
        echo "Monitor progress with: tail -f pipeline4_*.log"
        echo ""
        
        # Wait for all background processes to complete
        failed_datasets=()
        for i in "${!pids[@]}"; do
            pid="${pids[$i]}"
            dataset="${datasets[$i]}"
            
            echo "Waiting for $dataset (PID $pid)..."
            wait $pid
            exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                echo "$dataset completed successfully"
            else
                echo "$dataset failed with exit code $exit_code"
                failed_datasets+=("$dataset")
            fi
        done
        
        end_time=$(date +%s)
        total_duration=$((end_time - start_time))
        
        echo ""
        echo "=== PARALLEL EXECUTION COMPLETE ==="
        echo "Total time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
        
        if [ ${#failed_datasets[@]} -eq 0 ]; then
            echo "All datasets completed successfully"
        else
            echo "Failed datasets: ${failed_datasets[*]}"
            echo "Check logs: pipeline4_*.log"
        fi
        
        echo ""
        echo "Individual logs created:"
        for dataset in "${datasets[@]}"; do
            echo "  $dataset: pipeline4_${dataset}.log"
        done
    fi
    
else
    # Single dataset mode
    DATASET_NAME=$1
    
    # Validate dataset name
    if [[ ! " ${datasets[@]} " =~ " ${DATASET_NAME} " ]]; then
        echo "Error: Invalid dataset name '$DATASET_NAME'"
        echo "Valid datasets: ${datasets[*]}"
        show_help
        exit 1
    fi
    
    # Run single dataset
    run_single_dataset "$DATASET_NAME"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Dataset $DATASET_NAME completed successfully"
    else
        echo "Dataset $DATASET_NAME failed with exit code $exit_code"
    fi
    
    exit $exit_code
fi

echo ""
echo "=== SUMMARY ==="
echo "Check results in:"
echo "  Outputs: ${REPO_ROOT}/outputs/pipeline4/*"
echo "  Models:  ${REPO_ROOT}/models/pipeline4/*" 
echo "  Metrics: ${REPO_ROOT}/metrics/pipeline4/*"
echo "  Scores:  ${REPO_ROOT}/scores/pipeline4/*"

sleep 5

echo ""
echo "-- Pipeline 4 complete --"
echo "Results:"
echo "  • Statistical Analysis: ${REPO_ROOT}/Statistics/"
echo "  • Plots: ${REPO_ROOT}/plots/"
