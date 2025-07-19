# FILE: src/trainers/pretrain_trainer.py
"""
Bedrock Protocol: Main Pre-training Trainer.

This script orchestrates the end-to-end pre-training of a language model,
from loading data and model configuration to running the distributed training loop.
It integrates Hugging Face Accelerate and supports DeepSpeed for advanced parallelism.
"""

import sys
import os
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

# Ensure project root is in path for imports
# 修正: 找到项目根目录并添加到 sys.path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent # Assuming script is in src/trainers/
sys.path.append(str(project_root))

from src.models.language_model import BaseLLM


# Tokenizer is now loaded from HF format, so no direct `train_tokenizer` import needed here.

def get_total_params(model: torch.nn.Module) -> int:
    """Calculates the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_pretrain(config_path: str) -> None:
    """
    Main function to execute the pre-training process.

    Args:
        config_path: Path to the YAML configuration file.
    """
    print("--- [Bedrock] Initiating From-Scratch Pre-training ---")

    # 1. Load Configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize Accelerator
    # Mandate of Proactive Defense: Accelerate handles distributed setup robustly.
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        log_with=config['report_to'],
        project_dir=config['output_dir']
    )

    # Set seed for reproducibility across ranks
    set_seed(config['seed'])

    # 3. Experiment Tracking (WandB)
    if accelerator.is_main_process and config['report_to'] == "wandb":
        wandb.init(
            project="Awesome-LLM-ZeroToScratch",
            name=config['run_name'],
            config=config
        )
        print("WandB initialized.")

    # 4. Load Model Architecture Configuration
    # 修正: 使用 project_root 来构建模型配置路径
    model_config_path = project_root / config['model_config_path']
    print(f"Loading model architecture configuration from: {model_config_path}")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # 5. Load Tokenizer
    # Tokenizer is now saved in Hugging Face format by build_tokenizer.py
    # 修正: 使用 project_root 来构建分词器路径
    tokenizer_load_path = project_root / Path(model_config['tokenizer_path']).parent.name / (
                Path(model_config['tokenizer_path']).name + "_hf") # Correctly construct path to _hf dir

    # Fallback if the above path construction isn't perfectly right, try the exact path from config + _hf
    if not tokenizer_load_path.exists():
        tokenizer_load_path = Path(model_config['tokenizer_path'] + "_hf")
        if not tokenizer_load_path.exists():
            print(f"Error: Hugging Face format tokenizer not found at {tokenizer_load_path}.")
            print("Please ensure `data_processing/download_and_reproduce.py text` was run successfully.")
            sys.exit(1)


    print(f"Loading tokenizer from Hugging Face format: {tokenizer_load_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_load_path),
        trust_remote_code=True,
        use_fast=True  # Use fast tokenizer for performance
    )
    # Ensure padding token is set for consistent batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # 6. Load Dataset and Tokenize
    print(f"Loading dataset from: {config['dataset_dir']}")
    raw_dataset = load_from_disk(config['dataset_dir'])

    def tokenize_function(examples):
        # Handle cases where 'text' column might be a list or single string
        texts = examples[config['dataset_text_field']]
        if isinstance(texts, list) and all(isinstance(elem, list) for elem in texts):
            # If text_field contains lists of lists (e.g., VLM captions after processing)
            texts = [" ".join(sublist) for sublist in texts]
        elif isinstance(texts, list) and any(not isinstance(elem, str) for elem in texts):
            # Handle cases where some elements might not be strings
            texts = [str(elem) if elem is not None else "" for elem in texts]
        elif isinstance(texts, str):
            texts = [texts] # Wrap single string in a list for tokenizer

        # Filter out empty strings before tokenization to avoid errors
        filtered_texts = [t for t in texts if t.strip()]
        if not filtered_texts:
            return {"input_ids": [], "attention_mask": []} # Return empty if no valid text

        tokenized_output = tokenizer(
            filtered_texts, # 修正: 使用 filtered_texts
            max_length=config['max_seq_length'],
            truncation=True,
            padding="max_length"  # Pad to max_seq_length for uniform batching
        )
        return tokenized_output

    # Mandate of Empirical Proof: Tokenization is deterministic.
    # num_proc should be tuned based on CPU cores.
    num_processes_for_map = os.cpu_count() if os.cpu_count() is not None else 1
    print(f"Using {num_processes_for_map} processes for dataset tokenization.")

    # 修正: 调整 remove_columns 逻辑，确保正确处理 VLM 和 LLM 的列
    columns_to_remove = [col for col in raw_dataset['train'].column_names if col not in ['pixel_values', 'input_ids', 'attention_mask']]
    if config.get('is_vlm', False):
        # For VLM, keep 'pixel_values', 'input_ids', 'attention_mask', and potentially 'cleaned_captions'/'distilled_captions' for debug/info
        # If 'cleaned_captions' is the source of text, it should be removed after tokenization
        if config['dataset_text_field'] in columns_to_remove:
            columns_to_remove.remove(config['dataset_text_field']) # Ensure text field used for tokenization is removed
        # Ensure we don't remove other useful VLM columns if they are not 'pixel_values'
        vlm_specific_cols = ['processed_image_tensor', 'cleaned_captions', 'distilled_captions', 'is_valid']
        for col in vlm_specific_cols:
            if col in columns_to_remove:
                columns_to_remove.remove(col)
    else:
        # For LLM, remove the original text column after tokenization
        if config['dataset_text_field'] in raw_dataset['train'].column_names:
            columns_to_remove.append(config['dataset_text_field'])
        # Remove duplicates from columns_to_remove
        columns_to_remove = list(set(columns_to_remove))


    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove, # 修正：使用调整后的 remove_columns
        num_proc=num_processes_for_map,
        desc="Running tokenizer on dataset"
    )

    # 过滤掉 input_ids 为空的样本（例如，如果原始文本被过滤掉）
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        num_proc=num_processes_for_map,
        desc="Filtering empty tokenized examples"
    )


    # Create DataLoaders
    # num_workers can be optimized; for small demos, 0 or 4 is common.
    num_dataloader_workers = min(os.cpu_count() if os.cpu_count() is not None else 0, 4)
    print(f"Using {num_dataloader_workers} workers for DataLoaders.")

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=config['per_device_train_batch_size'],
        num_workers=num_dataloader_workers,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        shuffle=False,
        batch_size=config['per_device_train_batch_size'],  # Use same batch size for eval
        num_workers=num_dataloader_workers,
        pin_memory=True
    )
    print("Dataset tokenized and DataLoaders prepared.")

    # 7. Initialize Model
    print("Initializing model architecture...")
    model = BaseLLM(model_config)
    print(f"Model initialized with {get_total_params(model) / 1e9:.2f} Billion parameters.")

    # 8. Setup Optimizer and Scheduler
    print("Setting up optimizer and learning rate scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        eps=config['adam_epsilon']
    )

    num_training_steps = config['max_steps'] if config['max_steps'] > 0 else \
        len(train_dataloader) * config['num_train_epochs']

    lr_scheduler = get_scheduler(
        name=config['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * config['warmup_ratio']),
        num_training_steps=num_training_steps,
    )
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {int(num_training_steps * config['warmup_ratio'])}")

    # 9. Prepare for Distributed Training with Accelerate
    # This prepares model, optimizer, and data loaders for multi-GPU/distributed training.
    # It also applies DeepSpeed if `deepspeed_config` is set.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Apply torch.compile if enabled and available
    if config['use_torch_compile'] and hasattr(torch, 'compile'):
        # Only compile if not using DeepSpeed, as DeepSpeed handles its own graph optimizations.
        # Check if accelerator.state.deepspeed_plugin is none or deepspeed is not enabled
        if accelerator.state.deepspeed_plugin is None or not accelerator.state.deepspeed_plugin.deepspeed_config:
            print("Applying torch.compile to the model...")
            model = torch.compile(model)
        else:
            print("Skipping torch.compile because DeepSpeed is active.")

    print("Model, Optimizer, DataLoaders, and Scheduler prepared for training.")
    print(
        f"Effective batch size (per step): {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * accelerator.num_processes}")

    # 10. Training Loop
    print("\n--- Training Started ---")
    completed_steps = 0
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(config['num_train_epochs'] if config['num_train_epochs'] > 0 else 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Prepare inputs for the model (handle LLM vs VLM)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch.get('pixel_values', None)  # Only present for VLM

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask, # 修正: 传递 2D attention_mask
                pixel_values=pixel_values
            )
            logits = outputs['logits']
            aux_losses = outputs.get('aux_losses', {})

            # Compute loss (Standard Cross-Entropy Loss for language modeling)
            # Shift logits and labels for next token prediction
            # Logits correspond to text tokens after image tokens (if VLM)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()  # Labels are always based on original input_ids

            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id  # Ignore padding tokens from loss
            )

            # Add auxiliary losses (e.g., from MoE router)
            for k, v in aux_losses.items():
                loss += v  # Assume aux_losses are already scaled by their coefficients in the model

            # Backward pass and optimization
            # Mandate of Proactive Defense: Gradient accumulation handled by accelerator.
            accelerator.backward(loss)

            if (step + 1) % config['gradient_accumulation_steps'] == 0 or \
                    (step + 1) == len(train_dataloader):  # Last batch in epoch

                # Gradient clipping
                if config['max_grad_norm'] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                completed_steps += 1
                progress_bar.update(1)

                if accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr() if isinstance(lr_scheduler.get_last_lr(),
                                                                          list) else lr_scheduler.get_last_lr()
                    log_dict = {"train_loss": loss.item(), "learning_rate": current_lr}
                    log_dict.update({f"aux_loss/{k}": v.item() for k, v in aux_losses.items()})
                    accelerator.log(log_dict, step=completed_steps)
                    progress_bar.set_description(f"Loss: {loss.item():.4f}")

                if completed_steps % config['save_steps'] == 0:
                    output_path = Path(config['output_dir']) / f"step_{completed_steps}"
                    accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving
                    accelerator.save_state(output_path)  # Saves model, optimizer, scheduler state
                    if accelerator.is_main_process:
                        # For HF compatibility, save model weights and tokenizer separately
                        unwrapped_model = accelerator.unwrap_model(model)
                        model_save_path = output_path / "hf_model"
                        tokenizer_save_path = output_path / "hf_tokenizer"
                        model_save_path.mkdir(parents=True, exist_ok=True)
                        tokenizer_save_path.mkdir(parents=True, exist_ok=True)
                        # unwrapped_model.lm_head.weight = unwrapped_model.embed_tokens.weight  # Weight tying should be in __init__
                        torch.save(unwrapped_model.state_dict(), str(model_save_path / "pytorch_model.bin"))
                        tokenizer.save_pretrained(str(tokenizer_save_path))
                        print(f"Model and tokenizer checkpoint saved at step {completed_steps} to {output_path}")

                if config['max_steps'] > 0 and completed_steps >= config['max_steps']:
                    break  # Break if max_steps reached

        if config['max_steps'] > 0 and completed_steps >= config['max_steps']:
            break  # Break outer loop if max_steps reached

    progress_bar.close()
    print("--- Training Finished ---")

    # 11. Save Final Model (Accelerate handles saving for distributed models)
    if accelerator.is_main_process:
        final_model_path = Path(config['output_dir']) / "final_model"
        accelerator.wait_for_everyone()  # Ensure all processes are synchronized

        # Save the final model state and tokenizer in HF compatible format
        unwrapped_model = accelerator.unwrap_model(model)
        final_model_path.mkdir(parents=True, exist_ok=True)
        # unwrapped_model.lm_head.weight = unwrapped_model.embed_tokens.weight  # Weight tying should be in __init__
        torch.save(unwrapped_model.state_dict(), str(final_model_path / "pytorch_model.bin"))
        tokenizer.save_pretrained(str(final_model_path))  # Save tokenizer to the same dir

        print(f"Final model and tokenizer saved to {final_model_path}")
        if config['report_to'] == "wandb":
            wandb.finish()

    print("\n--- [Bedrock] Pre-training Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/pretrain_trainer.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1] # 修正: 获取正确的命令行参数
    run_pretrain(config_file_path)

# END OF FILE: src/trainers/pretrain_trainer.py