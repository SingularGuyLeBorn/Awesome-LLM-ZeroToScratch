# FILE: src/trainers/grpo_trainer.py
"""
Bedrock Protocol: Generalized Reinforcement Learning with Proximal Optimization (GRPO) Trainer.

This script implements GRPO, a PPO-like algorithm that learns from multiple
generated completions per prompt. This version is refactored to align with the Bedrock
PPO trainer, featuring an aggressive memory-saving strategy with quantization.
"""

# [HARDCODED MIRROR] Force Hugging Face Hub downloads to go through a domestic mirror
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import yaml
import random
import time
import gc
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torch.quantization
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.hf_api import RepoFile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model


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


def load_dataset_robustly(repo_id: str, split: str, text_field: str):
    """[ULTIMATE DATA ENGINE V12.0] Intelligently validates and downloads datasets."""
    print(f"\n[Data Engine] Initializing for dataset '{repo_id}'.")
    # This is a simplified version of the robust loader for brevity.
    # The full logic can be pasted here if needed.
    try:
        dataset = load_dataset(repo_id, split=split, download_mode="reuse_dataset_if_exists")
        print(f"\n[Data Engine] Successfully loaded the '{split}' split.")
        # Ensure the text field exists
        if text_field not in dataset.column_names:
            raise ValueError(f"Dataset does not have the specified text field '{text_field}'")
        return dataset
    except Exception as e:
        print(f"--> FATAL: Failed to load dataset. Error: {e}")
        sys.exit(1)


def _calculate_logps(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Core log-probability calculation function. Does not control gradients."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    log_softmax_logits = F.log_softmax(shifted_logits / temperature, dim=-1)
    return torch.gather(log_softmax_logits, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)


def run_grpo(config_path: str):
    """Main function to execute the GRPO process with memory-saving optimizations."""
    print("--- [Bedrock] Initiating Generalized Reinforcement Learning with Proximal Optimization (GRPO) ---")
    print(f"--> NOTE: Hugging Face endpoint is set to: {os.environ.get('HF_ENDPOINT')}")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
    set_seed(config['seed'])

    if accelerator.is_main_process:
        print(f"\n[Accelerator] Environment prepared for {accelerator.num_processes} process(es).")

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], use_fast=config['use_fast_tokenizer'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if accelerator.is_main_process: print("Tokenizer loaded successfully.")

    if config.get('dataset_name'):
        dataset = load_dataset_robustly(config['dataset_name'], "train", config['dataset_text_field'])
        if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
            dataset = dataset.select(range(int(config['dataset_subset_size'])))
            if accelerator.is_main_process: print(f"--> Using a subset of {len(dataset)} samples for this run.")
    else:
        if accelerator.is_main_process: print(
            "\n[Data] No dataset_name found, using dummy PromptDataset for demonstration.")
        subset_size = int(config.get('dataset_subset_size', 32))
        dataset = PromptDataset(subset_size)
        if accelerator.is_main_process: print(f"--> Dummy dataset created with {len(dataset)} samples.")

    def collate_fn(batch: List[Dict[str, str]]):
        if config.get('dataset_name'):
            prompts = [item[config['dataset_text_field']] for item in batch]
        else:
            prompts = [item["prompt"] for item in batch]
        return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    dataloader = DataLoader(dataset, batch_size=config['per_device_train_batch_size'], collate_fn=collate_fn,
                            shuffle=True)
    if accelerator.is_main_process: print(f"Dataset loaded and formatted with {len(dataset)} prompts.")

    # --- THIS ENTIRE BLOCK IS REWRITTEN TO FIX THE GRADIENT ISSUE ---
    use_quantization = config.get('quantize_models', False)
    print(
        f"\n[Model Loading] Extreme memory optimization enabled. Quantization: {'ON' if use_quantization else 'OFF'}.")
    peft_config_for_base = PeftConfig.from_pretrained(config['model_name_or_path'])
    base_model_name = peft_config_for_base.base_model_name_or_path

    print("--> Step 1: Loading SFT model and merging adapter...")
    sft_model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    sft_model_merged = PeftModel.from_pretrained(sft_model_base, config['model_name_or_path'])
    sft_model_merged = sft_model_merged.merge_and_unload()
    print("--> SFT adapter merged into a temporary base model.")
    del sft_model_base;
    gc.collect()

    print("--> Step 2a: Creating Reference model (as a frozen copy of the merged SFT model)...")
    ref_model = sft_model_merged if config['beta'] > 0.0 else None
    if ref_model:
        if use_quantization:
            print("--> Step 2b: Quantizing Reference model (int8)...")
            ref_model = torch.quantization.quantize_dynamic(ref_model.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8)
            print("--> Reference model quantized for memory saving.")
        ref_model.eval()

    print("--> Step 3: Creating TRAINABLE Policy model by applying a NEW LoRA adapter...")
    lora_config_grpo = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # The policy model uses the UN-QUANTIZED sft_model_merged instance
    model = get_peft_model(sft_model_merged, lora_config_grpo)
    print("--> Policy model with new, trainable LoRA adapter is ready.")
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    gc.collect()
    print("--> Step 4: All models initialized.")

    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
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

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.eval()
                with torch.no_grad():
                    gen_config = GenerationConfig(max_new_tokens=config['max_new_tokens'],
                                                  min_new_tokens=config['min_new_tokens'], top_k=0, top_p=1.0,
                                                  do_sample=True, pad_token_id=tokenizer.pad_token_id,
                                                  eos_token_id=tokenizer.eos_token_id)
                    prompt_completion_ids = unwrapped_model.generate(input_ids=repeated_prompt_ids,
                                                                     attention_mask=repeated_prompt_mask,
                                                                     generation_config=gen_config)
                unwrapped_model.train()

                attention_mask = (prompt_completion_ids != tokenizer.pad_token_id).long()
                completion_mask = torch.zeros_like(attention_mask);
                completion_mask[:, prompt_len:] = attention_mask[:, prompt_len:]

                rewards_per_func_local = torch.rand(len(repeated_prompt_ids), len(config['reward_funcs']),
                                                    device=accelerator.device)

                with torch.no_grad():
                    old_logits = model(prompt_completion_ids, attention_mask=attention_mask).logits
                    old_per_token_logps = _calculate_logps(old_logits, prompt_completion_ids, config['temperature'])
                    if ref_model:
                        ref_logits = ref_model(prompt_completion_ids, attention_mask=attention_mask).logits
                        ref_per_token_logps = _calculate_logps(ref_logits, prompt_completion_ids, config['temperature'])
                    else:
                        ref_per_token_logps = None

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
                    per_token_logps = _calculate_logps(logits, prompt_completion_ids, config['temperature'])

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