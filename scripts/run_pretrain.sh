# FILE: scripts/run_pretrain.sh
#!/bin/bash
# Bedrock Protocol: One-click script to launch the pre-training process.
# This script uses DeepSpeed, which is highly recommended for large-scale pre-training.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- [Bedrock] Launching Pre-training Trainer ---"

# --- Recommended Usage with DeepSpeed ---
# Before running, ensure you have configured Accelerate for DeepSpeed.
# You can do this by running:
# accelerate config
# And selecting DeepSpeed as your distributed training method.
# For example:
# - What processor(s) do you have? `all_cuda`
# - What type of machine are you using? `multi-GPU` (or `multi-node` if applicable)
# - How many processes in total in your distributed setup? (e.g., 4 for 4 GPUs)
# - Do you want to use DeepSpeed? `yes`
# - Do you want to use the BF16 mixed precision? `yes` (if your GPU supports it, e.g., A100, RTX 30/40 series)
# - What DeepSpeed config do you want to use? `all` (for ZeRO-3) or `ZeRO-2`
# - You can also provide a custom DeepSpeed config JSON path.

# Command to run the pre-training script using accelerate.
# It automatically detects your configured DeepSpeed setup.
accelerate launch src/trainers/pretrain_trainer.py configs/training/pretrain_llm.yaml

echo "--- [Bedrock] Pre-training script finished. ---"
# END OF FILE: scripts/run_pretrain.sh