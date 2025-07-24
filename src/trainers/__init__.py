# FILE: src/trainers/__init__.py
# Bedrock: This file makes the directory a Python package.

# Mandate of Zero Ambiguity: Expose key trainer functions for direct import.
from .pretrain_trainer import run_pretrain
from .sft_trainer import run_sft
from .dpo_trainer import run_dpo
from .ppo_trainer import run_ppo
from .grpo_trainer import run_grpo

# END OF FILE: src/trainers/__init__.py