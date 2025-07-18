# FILE: scripts/run_dpo.sh
#!/bin/bash
# Bedrock Protocol: One-click script to launch the DPO process.
set -e

echo "--- [Bedrock] Launching DPO Trainer ---"

# Ensure the SFT model path in the DPO config is correct before running.
# The script assumes the SFT training has already been completed.

accelerate launch src/trainers/dpo_trainer.py configs/training/finetune_dpo.yaml

echo "--- [Bedrock] DPO script finished. ---"
# END OF FILE: scripts/run_dpo.sh