#!/bin/bash

# Statistical Analysis and Visualization Script
# This script runs statistical analysis and generates plots for both disc and cup segmentation
# Can be called from any driver script after evaluation is complete
#
# Usage: ./analyze_and_plot.sh

# Activate virtual environment if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    :  # Already in a virtual environment
else
    echo "Warning: No virtual environment found."
fi

echo ""
echo "=== STATISTICAL ANALYSIS AND VISUALIZATION ==="
echo ""

sleep 5

echo "Running statistical analysis for DISC segmentation..."
python3 ${REPO_ROOT}/engine/statistical_analysis.py \
    --eval_type disc \
    --input_dir ${REPO_ROOT}/scores/disc \
    --output_dir ${REPO_ROOT}/Statistics/disc \
    --skip-summaries

sleep 5

echo ""
echo "Running statistical analysis for CUP segmentation..."
python3 ${REPO_ROOT}/engine/statistical_analysis.py \
    --eval_type cup \
    --input_dir ${REPO_ROOT}/scores/cup \
    --output_dir ${REPO_ROOT}/Statistics/cup \
    --skip-summaries

sleep 5

echo ""
echo "Generating comprehensive plots..."
python3 ${REPO_ROOT}/engine/plotting.py \
    --disc_results_dir ${REPO_ROOT}/Statistics/disc \
    --cup_results_dir ${REPO_ROOT}/Statistics/cup \
    --output_dir ${REPO_ROOT}/plots

echo ""
echo "=== ANALYSIS AND PLOTTING COMPLETE ==="
echo "Results:"
echo "  • Statistical Analysis: ${REPO_ROOT}/Statistics/"
echo "  • Plots: ${REPO_ROOT}/plots/"

