# FILE: scripts/run_ppo.sh
#!/bin/bash
# Bedrock Protocol: One-click script to launch the PPO process (Conceptual).
set -e

echo "--- [Bedrock] Launching PPO Trainer (Conceptual) ---"

# This script runs the PPO trainer. The configuration file points to the SFT
# model as its starting point. Ensure SFT has been run successfully first.
# The trainer script is adapted to run on CPU if no GPU is available.

accelerate launch src/trainers/ppo_trainer.py configs/training/finetune_ppo.yaml

echo "--- [Bedrock] PPO script finished. ---"

# END OF FILE: scripts/run_ppo.sh