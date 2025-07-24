# FILE: src/trainers/dpo_trainer.py
"""
Bedrock Protocol: Direct Preference Optimization (DPO) Trainer.

This script uses the Hugging Face TRL library to perform DPO, a form of
reinforcement learning from human feedback (RLHF) that is more stable and
computationally efficient than traditional PPO.
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
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, PeftConfig
from trl import DPOTrainer, DPOConfig


class RichDpoLogCallback(TrainerCallback):
    """
    A callback that prints DPO training logs in a clean, readable table format.
    """

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict, **kwargs):
        if state.is_world_process_zero and logs:
            # Filter out non-metric keys
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if not metrics:
                return

            # Header for the log table
            header = f"--- Step {state.global_step}/{state.max_steps} ---"
            separator = "=" * len(header)
            print(f"\n{header}")

            # Prepare rows for pretty printing
            log_items = [
                ("Loss", metrics.get("loss")),
                ("Learning Rate", metrics.get("learning_rate")),
                ("Rewards Chosen", metrics.get("rewards/chosen")),
                ("Rewards Rejected", metrics.get("rewards/rejected")),
                ("Accuracy", metrics.get("rewards/accuracies")),
                ("Margin", metrics.get("rewards/margins")),
            ]

            # Find the longest key for alignment
            max_key_len = max(len(key) for key, _ in log_items)

            for key, value in log_items:
                if value is not None:
                    print(f"{key:<{max_key_len}} : {value:.6f}")

            print(separator)


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


# Factory function for data formatting
def format_dpo_dataset_factory(tokenizer):
    def format_dpo_dataset(example: dict) -> dict:
        prompt_messages = example['chosen'][:-1]
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        chosen_str = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        rejected_str = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
        }

    return format_dpo_dataset


def run_dpo(config_path: str) -> None:
    print("--- [Bedrock] Initiating Direct Preference Optimization (DPO) ---")
    print(f"--> NOTE: Hugging Face endpoint is set to: {os.environ.get('HF_ENDPOINT')}")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    dataset = load_dataset_robustly(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        subset_size = int(config['dataset_subset_size'])
        dataset = dataset.select(range(subset_size))
        print(f"--> Using a subset of {subset_size} samples for this run.")

    formatting_function = format_dpo_dataset_factory(tokenizer)
    dataset = dataset.map(formatting_function, remove_columns=dataset.column_names)
    print(f"Dataset loaded and formatted with {len(dataset)} samples.")

    print(f"\n[Model Loading] Loading base SFT model for DPO: {config['model_name_or_path']}")
    print("--> Using 'force load into RAM' strategy for robust CPU execution.")

    peft_config_for_base = PeftConfig.from_pretrained(config['model_name_or_path'])
    base_model_name = peft_config_for_base.base_model_name_or_path

    print(f"--> Loading base model '{base_model_name}' fully into RAM...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    print("--> Loading SFT PEFT adapter and merging...")
    model = PeftModel.from_pretrained(base_model, config['model_name_or_path'])
    model = model.merge_and_unload()
    model.config.use_cache = False
    print("SFT-merged model prepared. This will be the base for DPO LoRA training.")

    model.tokenizer = tokenizer

    print("--> Setting `ref_model` to None as required for PEFT-based DPO training.")
    ref_model = None

    gc.collect()

    print("\n[Configuration] Configuring PEFT with LoRA for DPO training...")
    peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("[Configuration] Setting up training arguments using DPOConfig...")
    training_args = DPOConfig(
        output_dir=config['output_dir'],
        num_train_epochs=int(config['num_train_epochs']),
        per_device_train_batch_size=int(config['per_device_train_batch_size']),
        per_device_eval_batch_size=int(config['per_device_eval_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        optim=config['optim'],
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config.get('weight_decay', 0.0)),
        bf16=False,
        fp16=False,
        max_grad_norm=float(config['max_grad_norm']),
        logging_steps=int(config['logging_steps']),
        max_steps=int(config['max_steps']),
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to=config['report_to'],
        run_name=config['run_name'],
        remove_unused_columns=False,
        beta=float(config['beta']),
        max_prompt_length=int(config['max_prompt_length']),
        max_length=int(config['max_length']),
        max_completion_length=int(config.get('max_target_length', config['max_length'] - config['max_prompt_length']))
    )

    print("\n[Trainer Init] Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[RichDpoLogCallback],  # 关键优化：添加自定义日志回调
    )

    print("\n--- DPO Training Started ---")
    dpo_trainer.train()
    print("\n--- DPO Training Finished ---")

    final_model_path = Path(config['output_dir']) / "final_model"
    os.makedirs(final_model_path, exist_ok=True)

    print(f"\n[Saving] Saving final DPO-tuned adapter model to {final_model_path}...")
    dpo_trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print("Model and tokenizer saved successfully.")

    print("\n--- [Bedrock] DPO Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/dpo_trainer.py <path_to_config.yaml>")
        sys.exit(1)
    config_file_path = sys.argv[1]
    run_dpo(config_file_path)

# END OF FILE: src/trainers/dpo_trainer.py