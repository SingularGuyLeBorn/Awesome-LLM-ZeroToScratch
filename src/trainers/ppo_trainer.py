# FILE: src/trainers/ppo_trainer.py
"""
Bedrock Protocol: Proximal Policy Optimization (PPO) Trainer from Scratch.

This is a complete from-scratch implementation of PPO, designed with an
aggressive sequential memory management strategy to ensure it runs on
resource-constrained systems. This version is refactored to align with the
procedural style of the Bedrock DPO trainer.
"""

# [HARDCODED MIRROR] Force Hugging Face Hub downloads to go through a domestic mirror
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import yaml
import gc
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.quantization
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.hf_api import RepoFile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


# --- Helper Models & Utilities (Adapted from original PPO script) ---

class ActorModel(nn.Module):
    """Wrapper for the policy model to ensure consistent API."""

    def __init__(self, model: PeftModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
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


def compute_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      response_mask: torch.Tensor) -> torch.Tensor:
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


def compute_advantages_and_returns(rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor,
                                   gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes advantages and returns using Generalized Advantage Estimation (GAE)."""
    seq_len = rewards.size(1)
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    values_detached = values.detach()
    for t in reversed(range(seq_len - 1)):
        mask_t_plus_1 = response_mask[:, t + 1]
        next_values = values_detached[:, t + 1]
        effective_next_values = next_values * mask_t_plus_1
        delta = rewards[:, t] + gamma * effective_next_values - values_detached[:, t]
        last_gae_lam = delta + gamma * lam * last_gae_lam * mask_t_plus_1
        advantages[:, t] = last_gae_lam
    returns = advantages + values_detached[:, :-1]
    advantages = advantages * response_mask[:, 1:]
    return advantages, returns


def print_ppo_stats(epoch, total_epochs, total_loss, policy_loss, value_loss):
    """A dedicated function to print PPO training statistics."""
    header = f"    --- Epoch {epoch}/{total_epochs} ---"
    separator = "    " + "-" * (len(header) - 4)
    print(header)
    print(f"    Total Loss   : {total_loss:.6f}")
    print(f"    Policy Loss  : {policy_loss:.6f}")
    print(f"    Value Loss   : {value_loss:.6f}")
    print(separator)


def load_dataset_robustly(repo_id: str, split: str):
    """
    [ULTIMATE DATA ENGINE V12.0] Intelligently validates and downloads datasets.
    This function is adapted from the DPO trainer for consistency.
    """
    print(f"\n[Data Engine] Initializing for dataset '{repo_id}'.")
    print("--> Step 1/4: Performing pre-flight check of local cache...")
    local_cache_dir = Path(HF_HUB_CACHE) / f"datasets--{repo_id.replace('/', '--')}"
    is_complete = False
    try:
        api = HfApi()
        repo_files_info = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        def get_filename(file_info):
            return file_info.rfilename if isinstance(file_info, RepoFile) else file_info

        relevant_files = {get_filename(f) for f in repo_files_info if get_filename(f).endswith(
            ('.json', '.jsonl', '.parquet', '.arrow', '.csv', '.txt', '.py')) or "dataset_info.json" in get_filename(
            f) or "README.md" in get_filename(f)}
        if not local_cache_dir.exists():
            print("--> STATUS: Local cache directory does not exist. Full download required.")
            files_to_download = list(relevant_files)
        else:
            snapshot_dir = local_cache_dir / 'snapshots'
            if not snapshot_dir.exists():
                print("--> STATUS: Local cache directory exists but is empty. Full download required.")
                files_to_download = list(relevant_files)
            else:
                local_files_in_snapshot = {p.name for p in snapshot_dir.rglob('*') if p.is_file()}
                is_missing = any(
                    Path(f).name not in local_files_in_snapshot for f in relevant_files if not Path(f).is_dir())
                if not is_missing:
                    print("--> STATUS: Cache check passed. All files appear to be present. Skipping download.")
                    is_complete = True
                    files_to_download = []
                else:
                    print(f"--> STATUS: Cache incomplete. Full re-download will be triggered for safety.")
                    files_to_download = list(relevant_files)
    except Exception as e:
        print(f"--> WARNING: Pre-flight check failed. Assuming full download is needed. Error: {e}")
        api = HfApi()
        repo_files_info = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        def get_filename(file_info):
            return file_info.rfilename if isinstance(file_info, RepoFile) else file_info

        files_to_download = [get_filename(info) for info in repo_files_info if get_filename(info).endswith(
            ('.json', '.jsonl', '.parquet', '.arrow', '.csv', '.txt', '.py')) or "dataset_info.json" in get_filename(
            info) or "README.md" in get_filename(info)]
    if not is_complete:
        print(f"\n--> Step 2/4: Starting intelligent download of {len(files_to_download)} file(s)...")
        max_retries = 5
        initial_wait_time = 2
        for i, filename in enumerate(files_to_download):
            for attempt in range(max_retries):
                try:
                    print(
                        f"    - Downloading file {i + 1}/{len(files_to_download)}: '{filename}' (Attempt {attempt + 1}/{max_retries})...")
                    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", resume_download=True)
                    print(f"    - Successfully downloaded '{filename}'.")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = initial_wait_time * (2 ** attempt)
                        print(f"    - FAILED to download '{filename}'. Error: {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"    - FATAL: Failed to download '{filename}' after {max_retries} attempts.")
                        raise e
        print("--> Intelligent download complete.")
    try:
        print(f"\n--> Step 3/4: Loading dataset '{repo_id}' from local cache...")
        dataset = load_dataset(repo_id, split=split, download_mode="reuse_dataset_if_exists")
        print(f"\n[Data Engine] Successfully loaded the '{split}' split.")
        print("--> Step 4/4: Data Engine finished.")
        return dataset
    except Exception as e:
        print(
            f"--> FATAL: Failed to load dataset from cache even after download. Cache might be severely corrupted. Error: {e}")
        sys.exit(1)


def run_ppo(config_path: str) -> None:
    """Main function to execute the from-scratch PPO process."""
    print("--- [Bedrock] Initiating Proximal Policy Optimization (PPO) ---")
    print(f"--> NOTE: Hugging Face endpoint is set to: {os.environ.get('HF_ENDPOINT')}")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer loaded successfully.")

    dataset = load_dataset_robustly(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        subset_size = int(config['dataset_subset_size'])
        dataset = dataset.select(range(subset_size))
        print(f"--> Using a subset of {subset_size} samples for this run.")

    def tokenize_prompts(example: dict) -> dict:
        prompt = f"Review: {example[config['dataset_text_field']]}\nSentiment: "
        return tokenizer(prompt, truncation=True, max_length=128, padding='max_length')

    dataset = dataset.map(tokenize_prompts, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Dataset loaded and formatted with {len(dataset)} prompts.")

    print(f"\n[Model Loading] Extreme memory optimization with QUANTIZATION enabled for CPU.")
    peft_config_for_base = PeftConfig.from_pretrained(config['model_name_or_path'])
    base_model_name = peft_config_for_base.base_model_name_or_path
    print("--> Step 1a: Creating Value (Critic) model on CPU...")
    base_model_for_critic = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    critic = CriticModel(base_model_for_critic).to(device)
    print("--> Value model created.")
    print("--> Step 1b: Quantizing Critic's base model (int8)...")
    critic.model = torch.quantization.quantize_dynamic(critic.model.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8)
    print("--> Critic's base model quantized. This significantly reduces its memory footprint.")
    del base_model_for_critic
    gc.collect()
    print("--> Temporary base model for critic destroyed.")
    print("--> Step 2: Loading SFT model to serve as the base for both Actor and Reference...")
    sft_model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    sft_model_merged = PeftModel.from_pretrained(sft_model_base, config['model_name_or_path'])
    sft_model_merged = sft_model_merged.merge_and_unload()
    print("--> SFT adapter merged into a single model.")
    gc.collect()
    print("--> Step 3a: Creating Reference model (from merged SFT)...")
    ref_model = sft_model_merged
    for param in ref_model.parameters(): param.requires_grad = False
    ref_model.to(device)
    print("--> Reference model ready and frozen.")
    print("--> Step 3b: Quantizing Reference model (int8)...")
    ref_model = torch.quantization.quantize_dynamic(ref_model.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8)
    print("--> Reference model quantized. This is a major memory saving.")
    gc.collect()
    print("--> Step 4: Creating Policy (Actor) model by applying a new LoRA adapter...")
    lora_config_ppo = LoraConfig(r=int(config['lora_r']), lora_alpha=int(config['lora_alpha']),
                                 lora_dropout=float(config['lora_dropout']),
                                 target_modules=config['lora_target_modules'], bias="none", task_type="CAUSAL_LM")
    policy_peft_model = get_peft_model(sft_model_merged, lora_config_ppo)
    actor = ActorModel(policy_peft_model).to(device)
    print("--> Policy model (float32) with new LoRA adapter is ready for training.")
    print("--> Step 5: All models initialized with minimized peak memory usage for CPU.")

    trainable_params = [*filter(lambda p: p.requires_grad, actor.parameters()),
                        *filter(lambda p: p.requires_grad, critic.parameters())]
    optimizer = AdamW(trainable_params, lr=float(config['learning_rate']))
    print("\n[Optimizer] AdamW optimizer configured for LoRA and value head parameters.")

    print("\n--- PPO Training Started (From Scratch) ---")
    ppo_epochs, mini_batch_size, kl_coef, vf_coef, clip_epsilon, gamma, lam = int(config['ppo_epochs']), int(
        config['mini_batch_size']), float(config.get('init_kl_coef', 0.2)), float(config.get('vf_coef', 0.1)), float(
        config.get('cliprange', 0.2)), float(config.get('gamma', 0.99)), float(config.get('lam', 0.95))

    for ppo_step in range(int(config['ppo_steps'])):
        print(f"\n--- PPO Step {ppo_step + 1}/{config['ppo_steps']} ---")
        actor.eval()
        critic.eval()
        batch = next(iter(dataloader))
        prompt_ids, prompt_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        prompt_len = prompt_ids.size(1)

        print(f"  (1/4) Rollout: Generating responses...")
        response_ids = actor.generate(input_ids=prompt_ids, attention_mask=prompt_mask, min_length=-1, top_k=0.0,
                                      top_p=1.0, do_sample=True, pad_token_id=tokenizer.pad_token_id,
                                      max_new_tokens=int(config.get('max_new_tokens', 32)))
        full_ids, full_mask, response_only_ids = response_ids, (
                    response_ids != tokenizer.pad_token_id).long(), response_ids[:, prompt_len:]
        response_mask = torch.zeros_like(full_mask)
        response_mask[:, prompt_len:] = full_mask[:, prompt_len:]

        print(f"  (2/4) Evaluation: Calculating log_probs, values, and rewards...")
        with torch.no_grad():
            log_probs_policy = compute_log_probs(actor.model.to(device), full_ids, full_mask, response_mask)
            log_probs_ref = compute_log_probs(ref_model, full_ids, full_mask, response_mask)
            values = critic(input_ids=full_ids, attention_mask=full_mask)
            kl_div = log_probs_policy - log_probs_ref
            rewards = -kl_coef * kl_div
            decoded_responses = tokenizer.batch_decode(response_only_ids, skip_special_tokens=True)
            for i, resp in enumerate(decoded_responses):
                terminal_reward = len(set(resp.split())) / 10.0
                response_len = torch.sum(full_mask[i, prompt_len:]).int().item() - 1
                if response_len >= 0 and (prompt_len + response_len) < rewards.size(1): rewards[
                    i, prompt_len + response_len] += terminal_reward

        print(f"  (3/4) GAE: Computing advantages and returns...")
        advantages, returns = compute_advantages_and_returns(rewards, values, response_mask, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) if advantages.numel() > 1 else (
                    advantages - advantages.mean())

        print(f"  (4/4) Optimization: Updating policy and value models...")
        actor.train()
        critic.train()
        for epoch in range(ppo_epochs):
            perm = torch.randperm(full_ids.size(0))
            for i in range(0, full_ids.size(0), mini_batch_size):
                indices = perm[i:i + mini_batch_size]
                mb_ids, mb_mask, mb_log_probs_old, mb_returns, mb_advantages, mb_response_mask = full_ids[indices], \
                full_mask[indices], log_probs_policy[indices], returns[indices], advantages[indices], response_mask[
                    indices]
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
                policy_loss = (torch.max(policy_loss_1, policy_loss_2) * mb_response_mask[:,
                                                                         1:]).sum() / mb_response_mask[:, 1:].sum()
                value_loss = 0.5 * ((new_values - mb_returns) ** 2)
                value_loss = (value_loss * mb_response_mask[:, 1:]).sum() / mb_response_mask[:, 1:].sum()
                total_loss = policy_loss + vf_coef * value_loss
                total_loss.backward()
                optimizer.step()
            print_ppo_stats(epoch + 1, ppo_epochs, total_loss.item(), policy_loss.item(), value_loss.item())

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

# END OF FILE: src/trainers/ppo_trainer.py