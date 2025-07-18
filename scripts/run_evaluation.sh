# FILE: scripts/run_evaluation.sh

# Bedrock Protocol: Script to run model evaluation.
# This script is conceptual and demonstrates how to invoke evaluation.
# For full benchmarks, you'd typically use `lm-evaluation-harness` directly.
set -e

echo "--- [Bedrock] Launching Model Evaluation (Conceptual) ---"

# Usage: bash scripts/run_evaluation.sh <path_to_model_or_adapter> [tasks] [num_samples]

MODEL_PATH=${1:-"./checkpoints/sft-tinyllama-guanaco/final_model"} # Default to SFT model
TASKS=${2:-"mmlu_flan_n_shot"} # Default evaluation task
NUM_SAMPLES=${3:-10} # Default number of samples for quick conceptual run

echo "Model Path: $MODEL_PATH"
echo "Evaluation Tasks: $TASKS"
echo "Number of Samples (Conceptual): $NUM_SAMPLES"

# Execute the conceptual Python evaluation script
python src/evaluation/evaluate_llm.py "$MODEL_PATH" "$TASKS" "$NUM_SAMPLES"

echo "--- [Bedrock] Evaluation script finished. ---"
# To run full lm-evaluation-harness, you would typically use a command like:
# lm_eval --model hf --model_args pretrained=<your_model_path>,dtype=bfloat16 --tasks mmlu --device cuda --batch_size 4 --output_path ./evaluation_results.json
# END OF FILE: scripts/run_evaluation.sh