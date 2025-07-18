# FILE: scripts/run_inference.sh
#!/bin/bash
# Bedrock Protocol: Script to launch the model inference CLI.
set -e

echo "--- [Bedrock] Launching Model Inference CLI ---"

# Usage: bash scripts/run_inference.sh <path_to_model_or_adapter> [max_new_tokens]

MODEL_PATH=${1:-"./checkpoints/sft-tinyllama-guanaco/final_model"} # Default to SFT model
MAX_NEW_TOKENS=${2:-200} # Default max generation length

echo "Model Path: $MODEL_PATH"
echo "Max New Tokens: $MAX_NEW_TOKENS"

# Execute the Python inference script
python src/inference/inference.py "$MODEL_PATH" "$MAX_NEW_TOKENS"

echo "--- [Bedrock] Inference script finished. ---"
# END OF FILE: scripts/run_inference.sh