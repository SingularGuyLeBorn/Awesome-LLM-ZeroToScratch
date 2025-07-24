# FILE: src/trainers/grpo_trainer.py
"""
Bedrock Protocol: Generalized Reinforcement Learning with Proximal Optimization (GRPO) Trainer.

This script implements GRPO, a PPO-like algorithm that learns from multiple
generated completions per prompt. It is designed for multi-GPU training using
Hugging Face Accelerate and is refactored to align with the Bedrock trainer style.
"""

# [HARDCODED MIRROR] Force Hugging Face Hub downloads to go through a domestic mirror
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import yaml
import random
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.hf_api import RepoFile
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AdamW, set_seed, \
    GenerationConfig


# --- Helper Classes & Utilities ---

class PromptDataset(Dataset):
    """A dummy dataset that provides a list of prompts for demonstration."""

    def __init__(self, num_prompts: int):
        base_prompts = [f"Prompt for generation #{i + 1}:" for i in range(8)]
        self.prompts = (base_prompts * (num_prompts // len(base_prompts) + 1))[:num_prompts]

    def __len__(self) -> int: return len(self.prompts)

    def __getitem__(self, i: int) -> Dict[str, str]: return {"prompt": self.prompts[i]}


def print_grpo_stats(step, max_steps, loss, avg_reward):
    """A dedicated function to print GRPO training statistics."""
    header = f"--- Step {step}/{max_steps} ---"
    separator = "=" * len(header)
    print(f"\n{header}")
    print(f"{'Loss':<15} : {loss:.6f}")
    print(f"{'Avg Reward':<15} : {avg_reward:.4f}")
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
                    is_complete = True;
                    files_to_download = []
                else:
                    print(f"--> STATUS: Cache incomplete. Full re-download will be triggered for safety.");
                    files_to_download = list(relevant_files)
    except Exception as e:
        print(f"--> WARNING: Pre-flight check failed. Assuming full download is needed. Error: {e}");
        api = HfApi();
        repo_files_info = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        def get_filename(file_info):
            return file_info.rfilename if isinstance(file_info, RepoFile) else file_info

        files_to_download = [get_filename(info) for info in repo_files_info if get_filename(info).endswith(
            ('.json', '.jsonl', '.parquet', '.arrow', '.csv', '.txt', '.py')) or "dataset_info.json" in get_filename(
            info) or "README.md" in get_filename(info)]
    if not is_complete:
        print(f"\n--> Step 2/4: Starting intelligent download of {len(files_to_download)} file(s)...");
        max_retries = 5;
        initial_wait_time = 2
        for i, filename in enumerate(files_to_download):
            for attempt in range(max_retries):
                try:
                    print(
                        f"    - Downloading file {i + 1}/{len(files_to_download)}: '{filename}' (Attempt {attempt + 1}/{max_retries})...");
                    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", resume_download=True);
                    print(f"    - Successfully downloaded '{filename}'.");
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = initial_wait_time * (2 ** attempt); print(
                            f"    - FAILED to download '{filename}'. Error: {e}. Retrying in {wait_time} seconds..."); time.sleep(
                            wait_time)
                    else:
                        print(f"    - FATAL: Failed to download '{filename}' after {max_retries} attempts."); raise e
        print("--> Intelligent download complete.")
    try:
        print(f"\n--> Step 3/4: Loading dataset '{repo_id}' from local cache...");
        dataset = load_dataset(repo_id, split=split, download_mode="reuse_dataset_if_exists");
        print(f"\n[Data Engine] Successfully loaded the '{split}' split.");
        print("--> Step 4/4: Data Engine finished.");
        return dataset
    except Exception as e:
        print(
            f"--> FATAL: Failed to load dataset from cache even after download. Cache might be severely corrupted. Error: {e}");
        sys.exit(1)


def get_per_token_logps(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        logits_to_keep: int, temperature: float) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits, labels = logits[:, :-1, :], input_ids[:, 1:]
        logits, labels = logits[:, -logits_to_keep:], labels[:, -logits_to_keep:]
        log_softmax_logits = F.log_softmax(logits / temperature, dim=-1)
        return torch.gather(log_softmax_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def run_grpo(config_path: str):
    """Main function to execute the GRPO process."""
    print("--- [Bedrock] Initiating Generalized Reinforcement Learning with Proximal Optimization (GRPO) ---")
    print(f"--> NOTE: Hugging Face endpoint is set to: {os.environ.get('HF_ENDPOINT')}")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
    set_seed(config['seed'])

    if accelerator.is_main_process:
        print(f"\n[Accelerator] Environment prepared for {accelerator.num_processes} GPU(s).")

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], use_fast=config['use_fast_tokenizer'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if accelerator.is_main_process: print("Tokenizer loaded successfully.")

    if config.get('dataset_name'):
        dataset = load_dataset_robustly(config['dataset_name'], split="train")
    else:
        if accelerator.is_main_process: print(
            "\n[Data] No dataset_name found, using dummy PromptDataset for demonstration.")
        dataset = PromptDataset(config['num_prompts_for_demo'])

    def collate_fn(batch: List[Dict[str, str]]):
        prompts = [item["prompt"] for item in batch]
        return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    dataloader = DataLoader(dataset, batch_size=config['per_device_train_batch_size'], collate_fn=collate_fn,
                            shuffle=True)
    if accelerator.is_main_process: print(f"Dataset loaded and formatted with {len(dataset)} prompts.")

    print("\n[Model Loading] Loading base and reference models...")
    model = AutoModelForCausalLM.from_pretrained(config['model_name_or_path'])
    ref_model = AutoModelForCausalLM.from_pretrained(config['ref_model_name_or_path']) if config['beta'] > 0.0 else None
    if ref_model: ref_model.eval()
    if accelerator.is_main_process: print("Models loaded successfully.")

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    reward_weights = torch.tensor(config['reward_weights'], device=accelerator.device, dtype=torch.float32)

    model, ref_model, optimizer, dataloader = accelerator.prepare(model, ref_model, optimizer, dataloader)

    print("\n--- GRPO Training Started ---")
    total_steps, num_generations = 0, config['num_generations']
    while config['max_steps'] == -1 or total_steps < config['max_steps']:
        for batch in dataloader:
            if total_steps >= config['max_steps'] and config['max_steps'] != -1: break
            with accelerator.accumulate(model):
                prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
                prompt_len = prompt_ids.shape[1]
                repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
                repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

                model.eval()
                with torch.no_grad():
                    gen_config = GenerationConfig(max_new_tokens=config['max_new_tokens'],
                                                  min_new_tokens=config['min_new_tokens'],
                                                  top_k=config['generation_top_k'], top_p=config['generation_top_p'],
                                                  do_sample=config['generation_do_sample'],
                                                  pad_token_id=tokenizer.pad_token_id,
                                                  eos_token_id=tokenizer.eos_token_id)
                    prompt_completion_ids = accelerator.unwrap_model(model).generate(input_ids=repeated_prompt_ids,
                                                                                     attention_mask=repeated_prompt_mask,
                                                                                     generation_config=gen_config)
                model.train()

                completion_len = prompt_completion_ids.shape[1] - prompt_len
                attention_mask = (prompt_completion_ids != tokenizer.pad_token_id).long()
                completion_mask = torch.zeros_like(attention_mask);
                completion_mask[:, prompt_len:] = attention_mask[:, prompt_len:]

                rewards_per_func_local = torch.rand(len(repeated_prompt_ids), len(config['reward_funcs']),
                                                    device=accelerator.device)

                old_per_token_logps = get_per_token_logps(model, prompt_completion_ids, attention_mask, completion_len,
                                                          config['temperature'])
                ref_per_token_logps = get_per_token_logps(ref_model, prompt_completion_ids, attention_mask,
                                                          completion_len, config['temperature']) if ref_model else None

                rewards_per_func_global = accelerator.gather(rewards_per_func_local)
                rewards_global = (rewards_per_func_global * reward_weights.unsqueeze(0)).nansum(dim=1)

                grouped_rewards = rewards_global.view(-1, num_generations)
                mean_grouped_rewards = grouped_rewards.mean(dim=1).repeat_interleave(num_generations, dim=0)
                advantages = rewards_global - mean_grouped_rewards
                if config['scale_rewards']:
                    std_grouped_rewards = grouped_rewards.std(dim=1).repeat_interleave(num_generations, dim=0)
                    advantages = advantages / (std_grouped_rewards + 1e-8)

                process_slice = slice(accelerator.process_index * len(prompt_ids) * num_generations,
                                      (accelerator.process_index + 1) * len(prompt_ids) * num_generations)
                advantages_local = advantages[process_slice]

                for _ in range(config['num_iterations']):
                    logits = model(input_ids=prompt_completion_ids, attention_mask=attention_mask).logits
                    logits, labels = logits[:, :-1, :], prompt_completion_ids[:, 1:]
                    logits, labels = logits[:, -completion_len:], labels[:, -completion_len:]
                    log_softmax_logits = F.log_softmax(logits / config['temperature'], dim=-1)
                    per_token_logps = torch.gather(log_softmax_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                    ratio = torch.exp(per_token_logps - old_per_token_logps)
                    per_token_loss1 = ratio * advantages_local.unsqueeze(1)
                    per_token_loss2 = torch.clamp(ratio, 1.0 - config['epsilon_low'],
                                                  1.0 + config['epsilon_high']) * advantages_local.unsqueeze(1)
                    total_loss = -torch.min(per_token_loss1, per_token_loss2)

                    if ref_per_token_logps is not None:
                        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
                                    ref_per_token_logps - per_token_logps) - 1
                        total_loss = total_loss + config['beta'] * per_token_kl

                    loss = (total_loss * completion_mask[:, 1:]).sum() / completion_mask[:, 1:].sum().clamp(min=1.0)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

            total_steps += 1
            if total_steps % config['log_interval'] == 0:
                if accelerator.is_main_process:
                    print_grpo_stats(total_steps, config['max_steps'], loss.item(), rewards_global.mean().item())

    if accelerator.is_main_process:
        print("\n--- GRPO Training Finished ---")
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Saving] Saving final model to {output_dir}...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("Model and tokenizer saved successfully.")
    print("\n--- [Bedrock] GRPO Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: accelerate launch src/trainers/grpo_trainer.py <path_to_config.yaml>")
        sys.exit(1)
    config_file_path = sys.argv[1]
    run_grpo(config_file_path)

# END OF FILE: src/trainers/grpo_trainer.py