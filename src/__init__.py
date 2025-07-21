# FILE: src/__init__.py
# Bedrock: This file makes the directory a Python package.

# Mandate of Zero Ambiguity: Expose top-level components for direct import.
from .models.language_model import BaseLLM
from .trainers.pretrain_trainer import run_pretrain
from .trainers.sft_trainer import run_sft
from .trainers.dpo_trainer import run_dpo
from .trainers.ppo_trainer import run_ppo # Added PPO trainer
from .inference.inference import run_inference_cli
from .evaluation.evaluate_llm import run_evaluation

# END OF FILE: src/__init__.py