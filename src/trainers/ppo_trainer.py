# FILE: src/trainers/ppo_trainer.py
"""
Bedrock Protocol: Proximal Policy Optimization (PPO) Trainer (Conceptual).

This script provides a conceptual outline for using the Hugging Face TRL library
to perform PPO, a common reinforcement learning from human feedback (RLHF) algorithm.
"""

# [HARDCODED MIRROR] Force Hugging Face Hub downloads to go through a domestic mirror
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
from pathlib import Path
import yaml
import torch
import gc
import shutil
import time
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.hf_api import RepoFile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# Ensure project root is in path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def load_dataset_robustly(repo_id: str, split: str):
    """
    [ULTIMATE DATA ENGINE V12.0] Intelligently validates and downloads datasets.
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

        relevant_files = {
            get_filename(f) for f in repo_files_info
            if get_filename(f).endswith(('.json', '.jsonl', '.parquet', '.arrow', '.csv', '.txt',
                                         '.py')) or "dataset_info.json" in get_filename(
                f) or "README.md" in get_filename(f)
        }

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
    """
    Conceptual main function to execute the PPO process.
    """
    print("--- [Bedrock] Initiating Proximal Policy Optimization (PPO) (Conceptual) ---")
    print(f"--> NOTE: Hugging Face endpoint is set to: {os.environ.get('HF_ENDPOINT')}")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = load_dataset_robustly(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        subset_size = int(config['dataset_subset_size'])
        dataset = dataset.select(range(subset_size))
        print(f"--> Using a subset of {subset_size} samples for this run.")

    def format_ppo_dataset(example: dict) -> dict:
        return {"query": example[config['dataset_text_field']]}

    dataset = dataset.map(format_ppo_dataset, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize(example):
        return tokenizer(
            example["query"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "query"])
    print(f"Dataset loaded and formatted with {len(dataset)} prompts.")

    print(f"\n[Model Loading] Loading and preparing PPO Actor model from: {config['model_name_or_path']}")

    peft_config_for_base = PeftConfig.from_pretrained(config['model_name_or_path'])
    base_model_name = peft_config_for_base.base_model_name_or_path

    print("--> Step 1/2: Loading SFT-tuned model and merging into a clean base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    sft_merged_model = PeftModel.from_pretrained(base_model, config['model_name_or_path']).merge_and_unload()
    print("--> SFT weights merged successfully.")

    print("--> Step 2/2: Wrapping SFT model with ValueHead AND applying PEFT adapter in a single operation...")
    ppo_peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_merged_model,
        peft_config=ppo_peft_config,
    )
    print("--> PPO Actor model with ValueHead and LoRA adapter is ready.")

    # [FINAL FIX] The `print_trainable_parameters` method is part of the PEFT model,
    # but not the top-level wrapper. We can safely remove this line.
    # The trainer itself will log parameter counts upon initialization.
    # model.print_trainable_parameters()

    gc.collect()

    print("\n[Configuration] Setting up PPO training arguments...")

    log_with = config.get('report_to', 'none')
    if log_with == 'none':
        log_with = None

    ppo_config = PPOConfig(
        exp_name=config['run_name'],
        log_with=log_with,
        learning_rate=float(config['learning_rate']),
        batch_size=int(config['batch_size']),
        mini_batch_size=int(config['mini_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        target_kl=float(config.get('target_kl', 0.1)),
        adap_kl_ctrl=bool(config['adap_kl_ctrl']),
        seed=int(config['seed']),
        remove_unused_columns=False,
    )

    print("[Reward Model] Initializing conceptual Reward Model (rewards based on length)...")

    def get_dummy_reward(outputs_text):
        rewards = []
        for text in outputs_text:
            unique_words = len(set(text.split()))
            rewards.append(torch.tensor(float(unique_words)))
        return rewards

    print("\n[Trainer Init] Initializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("\n--- PPO Training Started (Conceptual) ---")
    generation_kwargs = GenerationConfig(
        min_length=-1,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=int(config['max_new_tokens']),
    )

    stats_keys_to_log = [
        "ppo/loss/total", "ppo/loss/policy", "ppo/loss/value",
        "ppo/returns/mean", "ppo/returns/var", "objective/kl", "ppo/policy/approxkl",
    ]

    total_ppo_steps = int(config['ppo_steps'])
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= total_ppo_steps:
            break

        query_tensors = batch['input_ids']
        queries = [q for q in query_tensors]

        response_tensors = ppo_trainer.generate(queries, **generation_kwargs.to_dict())

        batch['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = get_dummy_reward(batch["response"])

        stats = ppo_trainer.step(queries, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        log_output = f"Conceptual PPO Step {step + 1}/{total_ppo_steps}:"
        for key in stats_keys_to_log:
            value = stats.get(key)
            if value is not None:
                log_output += f" | {key.split('/')[-1]}: {float(value):.4f}"
        print(log_output)

    print("\n--- PPO Training Finished (Conceptual) ---")

    final_model_path = Path(config['output_dir']) / "final_ppo_model"
    os.makedirs(final_model_path, exist_ok=True)

    print(f"\n[Saving] Saving final PPO-tuned adapter model to {final_model_path}...")
    ppo_trainer.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("Model and tokenizer saved successfully.")

    print("\n--- [Bedrock] PPO Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/ppo_trainer.py <path_to_config.yaml>")
        sys.exit(1)
    config_file_path = sys.argv[1]
    run_ppo(config_file_path)