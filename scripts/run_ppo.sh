# FILE: scripts/run_ppo.sh
#!/bin/bash
# Bedrock Protocol: One-click script to launch the PPO process (Conceptual).
set -e

echo "--- [Bedrock] Launching PPO Trainer (Conceptual) ---"

# Before running, ensure you have configured Accelerate for your environment.
# accelerate config
# Make sure to point to your SFT model in configs/training/finetune_ppo.yaml

accelerate launch src/trainers/ppo_trainer.py configs/training/finetune_ppo.yaml

echo "--- [Bedrock] PPO script finished. ---"