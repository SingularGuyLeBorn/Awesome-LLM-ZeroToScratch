# FILE: src/trainers/sft_trainer.py
"""
Bedrock Protocol: Supervised Fine-Tuning (SFT) Trainer.

This script uses the Hugging Face TRL library to perform SFT on a language model
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. It is designed to be
driven by a YAML configuration file.
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
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


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


def run_sft(config_path: str) -> None:
    """
    Main function to execute the SFT process.
    """
    print("--- [Bedrock] Initiating Supervised Fine-Tuning (SFT) ---")
    print(f"--> NOTE: Hugging Face endpoint is set to: {os.environ.get('HF_ENDPOINT')}")

    print(f"\n[Configuration] Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = load_dataset_robustly(config['dataset_name'], split="train")

    if 'dataset_subset_size_cpu' in config and int(config['dataset_subset_size_cpu']) > 0:
        subset_size = int(config['dataset_subset_size_cpu'])
        print(f"--> Using a subset of {subset_size} samples for this run.")
        dataset = dataset.select(range(subset_size))

    print(f"Dataset loaded with {len(dataset)} samples.")

    print(f"\n[Model Loading] Loading base model for SFT: {config['model_name_or_path']}")

    device_map = {"": "cpu"}
    torch_dtype = torch.float32
    attn_implementation = "sdpa"

    print("--> No GPU detected. Configuring for CPU-only execution.")

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name_or_path'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 新版 trl 需要将 tokenizer 挂载到 model 对象上，以便 SFTTrainer 内部能自动找到它
    model.tokenizer = tokenizer

    print("Model and tokenizer loaded successfully.")
    gc.collect()

    print("\n[Configuration] Configuring PEFT with LoRA...")
    peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("PEFT configured.")

    print("[Configuration] Setting up training arguments...")
    # 使用 SFTConfig，它是 TrainingArguments 的子类，专门为 SFT 设计
    training_args = SFTConfig(
        output_dir=config['output_dir'],
        num_train_epochs=int(config['num_train_epochs']),
        per_device_train_batch_size=int(config['per_device_train_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        optim=config['optim'],
        save_steps=int(config['save_steps']),
        logging_steps=int(config['logging_steps']),
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        fp16=False,
        bf16=False,
        max_grad_norm=float(config['max_grad_norm']),
        max_steps=int(config['max_steps']),
        warmup_ratio=float(config['warmup_ratio']),
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to=config['report_to'],
        run_name=config['run_name'],
        # 关键修复：将 max_seq_length 移动到 SFTConfig 中
        max_seq_length=int(config['max_seq_length']),
        # 关键修复：dataset_text_field 现在是 SFTConfig 的一部分
        dataset_text_field=config['dataset_text_field'],
        # packing 参数也是 SFTConfig 的一部分
        packing=True,
    )
    print("Training arguments set.")

    print("\n[Trainer Init] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print("\n--- SFT Training Started ---")
    trainer.train()
    print("\n--- SFT Training Finished ---")

    final_model_path = Path(config['output_dir']) / "final_model"
    # Ensure the parent directory exists before saving.
    os.makedirs(final_model_path, exist_ok=True)

    print(f"\n[Saving] Saving final adapter model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    # tokenizer 已经挂载在 model 上，但为了保险起见，我们还是手动保存一下
    tokenizer.save_pretrained(str(final_model_path))
    print("Model and tokenizer saved successfully.")

    print("\n--- [Bedrock] SFT Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/sft_trainer.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    run_sft(config_file_path)

# END OF FILE: src/trainers/sft_trainer.py