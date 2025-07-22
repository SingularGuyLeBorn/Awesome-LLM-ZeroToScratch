# FILE: scripts/run_evaluation.sh
#!/bin/bash
# Bedrock Protocol: Script to run model evaluation.
# This script runs our custom evaluation tool which performs qualitative checks
# and provides guidance for quantitative benchmarking.
set -e

echo "--- [Bedrock] Launching Model Evaluation ---"

# Usage: bash scripts/run_evaluation.sh <path_to_model_or_adapter>
# Example: bash scripts/run_evaluation.sh ./checkpoints/dpo-tinyllama-guanaco-cpu/final_model

# Default to the DPO model if no argument is provided, as it's the final stage.
MODEL_PATH=${1:-"./checkpoints/dpo-tinyllama-guanaco-cpu/final_model"}

echo "Evaluating Model Path: $MODEL_PATH"

# Execute the Python evaluation script
python src/evaluation/evaluate_llm.py "$MODEL_PATH"

echo "--- [Bedrock] Evaluation script finished. ---"

# END OF FILE: scripts/run_evaluation.sh