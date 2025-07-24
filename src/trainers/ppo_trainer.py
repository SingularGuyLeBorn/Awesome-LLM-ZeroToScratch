# Bedrock Protocol: Proximal Policy Optimization (PPO) Trainer from Scratch.
# ULTIMATE MEMORY-OPTIMIZED FINAL VERSION - REFACTORED
#
# This version refactors the original script into a PPOConfig and PPOTrainer class
# to imitate the structure of libraries like TRL, without changing any core logic.
# The aggressive sequential memory management strategy remains intact.

import os
import sys
import copy
import gc
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.quantization
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


# --- Helper Models (Remain the same logically) ---

class ActorModel(nn.Module):
    """Wrapper for the policy model to ensure consistent API."""

    def __init__(self, model: PeftModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


class CriticModel(nn.Module):
    """Wrapper for the value model with a value head."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        hidden_size = model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        values = self.value_head(last_hidden_state).squeeze(-1)
        return values


# --- Class 1: PPOConfig (模仿 TRL 的 PPOConfig) ---

class PPOConfig:
    """
    Configuration class for the PPOTrainer, initialized from a YAML file.
    """

    def __init__(self, **kwargs):
        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        print("\n[Configuration] PPOConfig initialized successfully.")

    @classmethod
    def from_yaml(cls, config_path: str):
        """Loads configuration from a YAML file."""
        print(f"\n[Configuration] Loading configuration from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# --- Class 2: PPOTrainer (模仿 TRL 的 PPOTrainer) ---

class PPOTrainer:
    """
    A from-scratch PPO Trainer with extreme memory optimization.
    """

    def __init__(self,
                 config: PPOConfig,
                 actor: ActorModel,
                 critic: CriticModel,
                 ref_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 dataset: Dataset):
        """
        Initializes the PPOTrainer.
        """
        print("\n--- [Bedrock] Initializing PPOTrainer ---")
        self.config = config
        self.actor = actor
        self.critic = critic
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hardware] Trainer will use device: {self.device}")

        # Move models to the correct device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.ref_model.to(self.device)

        # Optimizer setup
        trainable_params = [
            *filter(lambda p: p.requires_grad, self.actor.parameters()),
            *filter(lambda p: p.requires_grad, self.critic.parameters())
        ]
        self.optimizer = AdamW(trainable_params, lr=float(self.config.learning_rate))
        print("\n[Optimizer] AdamW optimizer configured for LoRA and value head parameters.")

        # Dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        print(f"Dataset loaded and formatted with {len(self.dataset)} prompts.")

        # Generation kwargs
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": int(self.config.max_new_tokens),
        }

    @staticmethod
    def _compute_log_probs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes log probabilities of tokens in the response."""
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
            masked_log_probs = token_log_probs * response_mask[:, 1:]
            return masked_log_probs

    @staticmethod
    def _compute_advantages_and_returns(
            rewards: torch.Tensor,
            values: torch.Tensor,
            response_mask: torch.Tensor,
            gamma: float,
            lam: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes advantages and returns using Generalized Advantage Estimation (GAE)."""
        seq_len = rewards.size(1)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        values_detached = values.detach()

        for t in reversed(range(seq_len - 1)):  # Corrected loop range
            mask_t_plus_1 = response_mask[:, t + 1]
            next_values = values_detached[:, t + 1]
            effective_next_values = next_values * mask_t_plus_1
            delta = rewards[:, t] + gamma * effective_next_values - values_detached[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam * mask_t_plus_1
            advantages[:, t] = last_gae_lam

        returns = advantages + values_detached[:, :-1]
        advantages = advantages * response_mask[:, 1:]
        return advantages, returns

    def train(self):
        """Main PPO training loop."""
        print("\n--- PPO Training Started (From Scratch) ---")
        global_step = 0

        for ppo_step in range(int(self.config.ppo_steps)):
            print(f"\n--- PPO Step {ppo_step + 1}/{self.config.ppo_steps} ---")

            self.actor.eval()
            self.critic.eval()

            # --- 1. Rollout Phase ---
            print("  (1/4) Rollout: Generating responses...")
            batch = next(iter(self.dataloader))
            prompt_ids = batch['input_ids'].to(self.device)
            prompt_mask = batch['attention_mask'].to(self.device)
            prompt_len = prompt_ids.size(1)

            with torch.no_grad():
                response_ids = self.actor.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    **self.generation_kwargs
                )
                full_ids = response_ids
                full_mask = (full_ids != self.tokenizer.pad_token_id).long()
                response_only_ids = full_ids[:, prompt_len:]
                response_mask = torch.zeros_like(full_mask)
                response_mask[:, prompt_len:] = full_mask[:, prompt_len:]

            # --- 2. Evaluation Phase ---
            print("  (2/4) Evaluation: Calculating log_probs, values, and rewards...")
            with torch.no_grad():
                log_probs_policy = self._compute_log_probs(self.actor.model, full_ids, full_mask, response_mask)
                log_probs_ref = self._compute_log_probs(self.ref_model, full_ids, full_mask, response_mask)
                values = self.critic(input_ids=full_ids, attention_mask=full_mask)
                kl_div = log_probs_policy - log_probs_ref
                rewards = -self.config.init_kl_coef * kl_div

                # Add terminal reward based on response diversity
                decoded_responses = self.tokenizer.batch_decode(response_only_ids, skip_special_tokens=True)
                for i, resp in enumerate(decoded_responses):
                    terminal_reward = len(set(resp.split())) / 10.0
                    response_len = torch.sum(full_mask[i, prompt_len:]).int().item() - 1
                    if response_len >= 0 and (prompt_len + response_len) < rewards.size(1):
                        rewards[i, prompt_len + response_len] += terminal_reward

            # --- 3. GAE Phase ---
            print("  (3/4) GAE: Computing advantages and returns...")
            advantages, returns = self._compute_advantages_and_returns(
                rewards, values, response_mask, self.config.gamma, self.config.lam
            )
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages - advantages.mean()

            # --- 4. Optimization Phase ---
            print("  (4/4) Optimization: Updating policy and value models...")
            self.actor.train()
            self.critic.train()

            for epoch in range(int(self.config.ppo_epochs)):
                perm = torch.randperm(full_ids.size(0))
                for i in range(0, full_ids.size(0), int(self.config.mini_batch_size)):
                    global_step += 1
                    indices = perm[i:i + int(self.config.mini_batch_size)]

                    # Mini-batch data
                    mb_ids = full_ids[indices]
                    mb_mask = full_mask[indices]
                    mb_log_probs_old = log_probs_policy[indices]
                    mb_returns = returns[indices]
                    mb_advantages = advantages[indices]
                    mb_response_mask = response_mask[indices]

                    self.optimizer.zero_grad()

                    # Actor loss
                    outputs = self.actor(input_ids=mb_ids, attention_mask=mb_mask)
                    logits = outputs.logits[:, :-1, :]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    labels = mb_ids[:, 1:]
                    new_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
                    log_ratio = new_log_probs - mb_log_probs_old
                    ratio = torch.exp(log_ratio)
                    policy_loss_1 = -mb_advantages * ratio
                    policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.config.cliprange,
                                                                 1.0 + self.config.cliprange)
                    policy_loss = (torch.max(policy_loss_1, policy_loss_2) * mb_response_mask[:,
                                                                             1:]).sum() / mb_response_mask[:, 1:].sum()

                    # Critic loss
                    new_values = self.critic(input_ids=mb_ids, attention_mask=mb_mask)[:, :-1]
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2)
                    value_loss = (value_loss * mb_response_mask[:, 1:]).sum() / mb_response_mask[:, 1:].sum()

                    # Total loss
                    total_loss = policy_loss + self.config.vf_coef * value_loss
                    total_loss.backward()
                    self.optimizer.step()

                print(
                    f"    Epoch {epoch + 1}/{self.config.ppo_epochs} | Total Loss: {total_loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")

        print("\n--- PPO Training Finished ---")

    def save_model(self, output_dir: Optional[str] = None):
        """Saves the final trained actor model and tokenizer."""
        if output_dir is None:
            output_dir = self.config.output_dir

        final_model_path = Path(output_dir) / "final_ppo_model_from_scratch"
        os.makedirs(final_model_path, exist_ok=True)

        print(f"\n[Saving] Saving final PPO-tuned adapter model to {final_model_path}...")
        self.actor.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        print("Model and tokenizer saved successfully.")


# --- Utility Functions (Data and Model Loading) ---

def load_dataset_robustly(repo_id: str, split: str):
    """Offline-first, robust data loader."""
    print(f"\n[Data Engine] Initializing for dataset '{repo_id}'.")
    try:
        print("--> Attempting to load directly from local cache (offline-first)...")
        dataset = load_dataset(repo_id, split=split, download_mode="reuse_dataset_if_exists")
        print("\n[Data Engine] Successfully loaded from cache.")
        return dataset
    except Exception as e:
        print(f"--> INFO: Could not load from cache directly. Will now attempt standard download. Error: {e}")
        dataset = load_dataset(repo_id, split=split)
        print(f"\n[Data Engine] Successfully loaded the '{split}' split.")
        return dataset


def create_models_and_tokenizer(config: PPOConfig) -> Tuple[
    ActorModel, CriticModel, PreTrainedModel, PreTrainedTokenizer]:
    """
    Handles the complex, memory-optimized model loading and quantization.
    This logic is identical to the original script.
    """
    print(f"\n[Model Loading] Extreme memory optimization with QUANTIZATION enabled for CPU.")

    device = torch.device("cpu")  # Load all models to CPU first
    model_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    peft_config_for_base = PeftConfig.from_pretrained(config.model_name_or_path)
    base_model_name = peft_config_for_base.base_model_name_or_path

    # Step 1: Create and Quantize Critic
    print("--> Step 1a: Creating Value (Critic) model on CPU...")
    base_model_for_critic = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype)
    critic = CriticModel(base_model_for_critic)  # Keep on CPU
    print("--> Value model created.")

    print("--> Step 1b: Quantizing Critic's base model (int8)...")
    critic.model = torch.quantization.quantize_dynamic(critic.model, {torch.nn.Linear}, dtype=torch.qint8)
    print("--> Critic's base model quantized.")
    del base_model_for_critic
    gc.collect()
    print("--> Temporary base model for critic destroyed.")

    # Step 2: Load SFT model for Actor and Reference
    print("--> Step 2: Loading SFT model to serve as the base for both Actor and Reference...")
    sft_model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype)
    sft_model_merged = PeftModel.from_pretrained(sft_model_base, config.model_name_or_path)
    sft_model_merged = sft_model_merged.merge_and_unload()  # This is now a base model
    print("--> SFT adapter merged into a single model.")
    gc.collect()

    # Step 3: Create and Quantize Reference Model
    print("--> Step 3a: Creating Reference model (from merged SFT)...")
    ref_model = copy.deepcopy(sft_model_merged)  # Create a copy for the ref model
    for param in ref_model.parameters():
        param.requires_grad = False
    print("--> Reference model ready and frozen.")

    print("--> Step 3b: Quantizing Reference model (int8)...")
    ref_model = torch.quantization.quantize_dynamic(ref_model, {torch.nn.Linear}, dtype=torch.qint8)
    print("--> Reference model quantized. This is a major memory saving.")
    gc.collect()

    # Step 4: Create Actor with new LoRA adapter
    print("--> Step 4: Creating Policy (Actor) model by applying a new LoRA adapter...")
    lora_config_ppo = LoraConfig(
        r=int(config.lora_r),
        lora_alpha=int(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # The actor uses the un-quantized sft_model_merged instance
    policy_peft_model = get_peft_model(sft_model_merged, lora_config_ppo)
    actor = ActorModel(policy_peft_model)
    print("--> Policy model (float32) with new LoRA adapter is ready for training.")

    print("--> Step 5: All models initialized with minimized peak memory usage for CPU.")

    return actor, critic, ref_model, tokenizer


# --- Main Execution Block ---

def main(config_path: str):
    """
    Main function to execute the refactored PPO process.
    """
    # 1. Load Configuration
    config = PPOConfig.from_yaml(config_path)

    # 2. Load and Prepare Models & Tokenizer using the memory-optimized factory
    actor, critic, ref_model, tokenizer = create_models_and_tokenizer(config)

    # 3. Load and Prepare Dataset
    raw_dataset = load_dataset_robustly(config.dataset_name, split="train")
    if hasattr(config, 'dataset_subset_size') and int(config.dataset_subset_size) > 0:
        subset_size = int(config.dataset_subset_size)
        raw_dataset = raw_dataset.select(range(subset_size))
        print(f"--> Using a subset of {subset_size} samples for this run.")

    def tokenize_prompts(example: dict) -> dict:
        prompt = f"Review: {example[config.dataset_text_field]}\nSentiment: "
        return tokenizer(prompt, truncation=True, max_length=128, padding='max_length')

    dataset = raw_dataset.map(tokenize_prompts, remove_columns=raw_dataset.column_names)
    dataset.set_format(type="torch")

    # 4. Initialize Trainer
    ppo_trainer = PPOTrainer(
        config=config,
        actor=actor,
        critic=critic,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset
    )

    # 5. Run Training
    ppo_trainer.train()

    # 6. Save Final Model
    ppo_trainer.save_model()

    print("\n--- [Bedrock] PPO Process from Scratch Complete ---")


if __name__ == "__main__":
    # You would need a config file named 'ppo_config.yaml' in the same directory
    # with the content you provided.
    # For example:
    # with open("ppo_config.yaml", "w") as f:
    #     f.write("""
    #     model_name_or_path: "./checkpoints/sft-tinyllama-guanaco-cpu/final_model"
    #     ... (rest of your yaml content)
    #     """)

    if len(sys.argv) != 2:
        print("Usage: python ppo_trainer_refactored.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    main(config_file_path)