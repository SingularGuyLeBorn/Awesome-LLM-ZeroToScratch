# Bedrock Protocol: Proximal Policy Optimization (PPO) Trainer from Scratch.
# ULTIMATE MEMORY-OPTIMIZED FINAL VERSION: This is a complete from-scratch implementation
# of PPO, designed with an aggressive sequential memory management strategy to ensure
# it runs on resource-constrained systems by minimizing peak memory usage, addressing
# the previous memory access violation error (exit status 3221225477).

import os
import sys
import copy
import gc
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.quantization  # Import the quantization module
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

# Ensure project root is in path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# --- Data Loading Utility (Remains the same) ---
def load_dataset_robustly(repo_id: str, split: str):
    """
    Offline-first, robust data loader.
    """
    print(f"\n[Data Engine] Initializing for dataset '{repo_id}'.")
    try:
        print("--> Attempting to load directly from local cache (offline-first)...")
        dataset = load_dataset(repo_id, split=split, download_mode="reuse_dataset_if_exists")
        print("\n[Data Engine] Successfully loaded from cache.")
        return dataset
    except Exception as e:
        print(f"--> INFO: Could not load from cache directly. Will now attempt standard download. Error: {e}")
        try:
            print(f"\n--> Loading dataset '{repo_id}' using standard procedure...")
            dataset = load_dataset(repo_id, split=split)
            print(f"\n[Data Engine] Successfully loaded the '{split}' split.")
            return dataset
        except Exception as final_e:
            print(
                f"--> FATAL: All attempts to load the dataset failed. Please check your network and cache. Error: {final_e}")
            sys.exit(1)


# --- Custom PPO Model and Utilities ---

class ActorModel(nn.Module):
    """Wrapper for the policy model to ensure consistent API."""

    def __init__(self, model: PeftModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class CriticModel(nn.Module):
    """Wrapper for the value model with a value head."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        # Add the value head
        hidden_size = model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        # Initialize the value head
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        # Pass it through the value head
        values = self.value_head(last_hidden_state).squeeze(-1)
        return values


def compute_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      response_mask: torch.Tensor) -> torch.Tensor:
    """Computes log probabilities of tokens in the response."""
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Shift logits and labels for next token prediction
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather the log probabilities of the actual tokens
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)

        # We only care about the log probabilities of the response tokens
        masked_log_probs = token_log_probs * response_mask[:, 1:]
        return masked_log_probs


def compute_advantages_and_returns(
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

    for t in reversed(range(seq_len)):
        mask_t_plus_1 = response_mask[:, t + 1] if t < seq_len else torch.zeros_like(response_mask[:, t])
        next_values = values_detached[:, t + 1] if t < seq_len else torch.zeros_like(values_detached[:, t])
        effective_next_values = next_values * mask_t_plus_1

        delta = rewards[:, t] + gamma * effective_next_values - values_detached[:, t]

        last_gae_lam = delta + gamma * lam * last_gae_lam * mask_t_plus_1
        advantages[:, t] = last_gae_lam

    returns = advantages + values_detached[:, :-1]

    advantages = advantages * response_mask[:, 1:]
    return advantages, returns


# --- Main PPO Trainer Logic ---

def run_ppo(config_path: str) -> None:
    """
    Main function to execute the from-scratch PPO process.
    """
    print("--- [Bedrock] Initiating PPO from Scratch ---")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Using device: {device}")

    raw_dataset = load_dataset_robustly(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        subset_size = int(config['dataset_subset_size'])
        raw_dataset = raw_dataset.select(range(subset_size))
        print(f"--> Using a subset of {subset_size} samples for this run.")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    def tokenize_prompts(example: dict) -> dict:
        prompt = f"Review: {example[config['dataset_text_field']]}\nSentiment: "
        return tokenizer(
            prompt,
            truncation=True,
            max_length=128,
            padding='max_length'
        )

    dataset = raw_dataset.map(tokenize_prompts, remove_columns=raw_dataset.column_names)
    dataset.set_format(type="torch")

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Dataset loaded and formatted with {len(dataset)} prompts.")

    # --- ULTIMATE MEMORY-OPTIMIZED Model Loading (CPU QUANTIZATION ENABLED) ---
    print(f"\n[Model Loading] Extreme memory optimization with QUANTIZATION enabled for CPU.")

    model_dtype = torch.float32

    peft_config_for_base = PeftConfig.from_pretrained(config['model_name_or_path'])
    base_model_name = peft_config_for_base.base_model_name_or_path

    # --- STRATEGY: Quantize non-trainable models (Critic base, Reference) ---

    print("--> Step 1a: Creating Value (Critic) model...")
    base_model_for_critic = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype)
    critic = CriticModel(base_model_for_critic).to(device)
    print("--> Value model created.")

    print("--> Step 1b: Quantizing Critic's base model (int8)...")
    critic.model = torch.quantization.quantize_dynamic(
        critic.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("--> Critic's base model quantized. This significantly reduces its memory footprint.")
    del base_model_for_critic
    gc.collect()
    print("--> Temporary base model for critic destroyed.")

    print("--> Step 2: Loading SFT model to serve as the base for both Actor and Reference...")
    sft_model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype)
    sft_model_merged = PeftModel.from_pretrained(sft_model_base, config['model_name_or_path'])
    sft_model_merged = sft_model_merged.merge_and_unload()
    sft_model_merged.to(device)
    print("--> SFT adapter merged into a single model.")
    gc.collect()

    print("--> Step 3a: Creating Reference model (no copy needed)...")
    ref_model = sft_model_merged
    for param in ref_model.parameters():
        param.requires_grad = False
    print("--> Reference model ready and frozen.")

    print("--> Step 3b: Quantizing Reference model (int8)...")
    ref_model = torch.quantization.quantize_dynamic(
        ref_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("--> Reference model quantized. This is a major memory saving.")
    gc.collect()

    print("--> Step 4: Creating Policy (Actor) model by applying a new LoRA adapter...")
    lora_config_ppo = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # The actor uses the SAME sft_model_merged instance which has NOT been quantized.
    # The LoRA layers will be applied to the float32 base weights.
    policy_peft_model = get_peft_model(sft_model_merged, lora_config_ppo)
    actor = ActorModel(policy_peft_model).to(device)
    print("--> Policy model (float32) with new LoRA adapter is ready for training.")

    print("--> Step 5: All models initialized with minimized peak memory usage for CPU.")

    trainable_params = [
        *filter(lambda p: p.requires_grad, actor.parameters()),
        *filter(lambda p: p.requires_grad, critic.parameters())
    ]
    optimizer = AdamW(trainable_params, lr=float(config['learning_rate']))
    print("\n[Optimizer] AdamW optimizer configured for LoRA and value head parameters.")

    print("\n--- PPO Training Started (From Scratch) ---")

    ppo_epochs = int(config['ppo_epochs'])
    mini_batch_size = int(config['mini_batch_size'])
    kl_coef = float(config.get('init_kl_coef', 0.2))
    vf_coef = float(config.get('vf_coef', 0.1))
    clip_epsilon = float(config.get('cliprange', 0.2))
    gamma = float(config.get('gamma', 0.99))
    lam = float(config.get('lam', 0.95))

    global_step = 0
    for ppo_step in range(int(config['ppo_steps'])):
        print(f"\n--- PPO Step {ppo_step + 1}/{config['ppo_steps']} ---")

        actor.eval()
        critic.eval()

        batch = next(iter(dataloader))
        prompt_ids = batch['input_ids'].to(device)
        prompt_mask = batch['attention_mask'].to(device)
        prompt_len = prompt_ids.size(1)

        print(f"  (1/4) Rollout: Generating responses...")
        with torch.no_grad():
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "max_new_tokens": int(config.get('max_new_tokens', 32)),
            }
            response_ids = actor.model.generate(input_ids=prompt_ids, attention_mask=prompt_mask, **generation_kwargs)

            full_ids = response_ids
            full_mask = (full_ids != tokenizer.pad_token_id).long()
            response_only_ids = full_ids[:, prompt_len:]

            response_mask = torch.zeros_like(full_mask)
            response_mask[:, prompt_len:] = full_mask[:, prompt_len:]

        print(f"  (2/4) Evaluation: Calculating log_probs, values, and rewards...")
        with torch.no_grad():
            log_probs_policy = compute_log_probs(actor.model, full_ids, full_mask, response_mask)
            log_probs_ref = compute_log_probs(ref_model, full_ids, full_mask, response_mask)
            values = critic(input_ids=full_ids, attention_mask=full_mask)
            kl_div = log_probs_policy - log_probs_ref
            rewards = -kl_coef * kl_div

            decoded_responses = tokenizer.batch_decode(response_only_ids, skip_special_tokens=True)
            for i, resp in enumerate(decoded_responses):
                terminal_reward = len(set(resp.split())) / 10.0
                response_len = torch.sum(full_mask[i, prompt_len:]).int().item() - 1
                if response_len >= 0 and (prompt_len + response_len) < rewards.size(1):
                    rewards[i, prompt_len + response_len] += terminal_reward

        print(f"  (3/4) GAE: Computing advantages and returns...")
        advantages, returns = compute_advantages_and_returns(rewards, values, response_mask, gamma, lam)
        # Handle case of std=0 when batch size is 1
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages - advantages.mean())

        rollout_data = {
            'full_ids': full_ids,
            'full_mask': full_mask,
            'log_probs_old': log_probs_policy,
            'returns': returns,
            'advantages': advantages,
            'response_mask': response_mask
        }

        print(f"  (4/4) Optimization: Updating policy and value models...")
        actor.train()
        critic.train()

        for epoch in range(ppo_epochs):
            perm = torch.randperm(full_ids.size(0))
            for i in range(0, full_ids.size(0), mini_batch_size):
                global_step += 1
                indices = perm[i:i + mini_batch_size]

                mb_ids = rollout_data['full_ids'][indices]
                mb_mask = rollout_data['full_mask'][indices]
                mb_log_probs_old = rollout_data['log_probs_old'][indices]
                mb_returns = rollout_data['returns'][indices]
                mb_advantages = rollout_data['advantages'][indices]
                mb_response_mask = rollout_data['response_mask'][indices]

                optimizer.zero_grad()

                outputs = actor(input_ids=mb_ids, attention_mask=mb_mask)
                logits = outputs.logits[:, :-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                labels = mb_ids[:, 1:]
                new_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)

                new_values = critic(input_ids=mb_ids, attention_mask=mb_mask)[:, :-1]

                log_ratio = new_log_probs - mb_log_probs_old
                ratio = torch.exp(log_ratio)

                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

                policy_loss = torch.max(policy_loss_1, policy_loss_2)
                policy_loss = (policy_loss * mb_response_mask[:, 1:]).sum() / mb_response_mask[:, 1:].sum()

                value_loss = 0.5 * ((new_values - mb_returns) ** 2)
                value_loss = (value_loss * mb_response_mask[:, 1:]).sum() / mb_response_mask[:, 1:].sum()

                total_loss = policy_loss + vf_coef * value_loss

                total_loss.backward()
                optimizer.step()

            print(
                f"    Epoch {epoch + 1}/{ppo_epochs} | Total Loss: {total_loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")

    print("\n--- PPO Training Finished ---")

    final_model_path = Path(config['output_dir']) / "final_ppo_model_from_scratch"
    os.makedirs(final_model_path, exist_ok=True)

    print(f"\n[Saving] Saving final PPO-tuned adapter model to {final_model_path}...")
    actor.model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("Model and tokenizer saved successfully.")
    print("\n--- [Bedrock] PPO Process from Scratch Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/ppo_trainer.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    run_ppo(config_file_path)