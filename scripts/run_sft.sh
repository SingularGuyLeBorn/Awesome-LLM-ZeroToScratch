# FILE: scripts/run_sft.sh
#!/bin/bash
# Bedrock Protocol: One-click script to launch the SFT process.
set -e

echo "--- [Bedrock] Launching SFT Trainer ---"

# We use `accelerate launch` which is the standard, robust way to handle
# single-node multi-GPU or multi-node training, managed by Hugging Face Accelerate.
# It correctly sets up the distributed environment.

accelerate launch src/trainers/sft_trainer.py configs/training/finetune_sft.yaml

echo "--- [Bedrock] SFT script finished. ---"
# END OF FILE: scripts/run_sft.sh