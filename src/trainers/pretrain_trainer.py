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
from datasets import load_from_disk
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

# Ensure project root is in path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.append(str(project_root))

from src.models.language_model import BaseLLM, BaseLLMConfig


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
    accelerator = Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        log_with=config.get('report_to'),
        project_dir=config['output_dir']
    )

    set_seed(int(config['seed']))

    # 3. Experiment Tracking (WandB)
    if accelerator.is_main_process and config.get('report_to') == "wandb":
        wandb.init(
            project="Awesome-LLM-ZeroToScratch",
            name=config['run_name'],
            config=config
        )
        print("WandB initialized.")

    # 4. Load Model Architecture Configuration
    model_config_path = project_root / config['model_config_path']
    print(f"Loading model architecture configuration from: {model_config_path}")
    with open(model_config_path, 'r') as f:
        model_config_dict = yaml.safe_load(f)

    model_config_dict['model_type_llm'] = model_config_dict.pop('model_type', 'DenseLLM')

    # 5. Load Tokenizer
    tokenizer_path_str = model_config_dict['tokenizer_path']
    hf_tokenizer_path_str = f"{tokenizer_path_str}_hf"
    tokenizer_load_path = project_root / hf_tokenizer_path_str
    if not tokenizer_load_path.exists():
        tokenizer_load_path = Path(tokenizer_path_str + "_hf")
    print(f"Loading tokenizer from Hugging Face format: {tokenizer_load_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_load_path), trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # 6. Load Dataset and Tokenize
    dataset_dir = project_root / config['dataset_dir']
    print(f"Loading dataset from: {dataset_dir}")
    raw_dataset = load_from_disk(str(dataset_dir))

    def tokenize_function(examples):
        return tokenizer(examples[config['dataset_text_field']], max_length=int(config['max_seq_length']),
                         truncation=True, padding="max_length")

    num_processes_for_map = min(os.cpu_count() or 1, 8)
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True,
                                        remove_columns=raw_dataset['train'].column_names,
                                        num_proc=num_processes_for_map, desc="Tokenizing dataset")
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True,
                                  batch_size=int(config['per_device_train_batch_size']))
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=int(config['per_device_train_batch_size']))
    print("Dataset tokenized and DataLoaders prepared.")

    # 7. Initialize Model
    print("Initializing model architecture...")
    model_config_dict['vocab_size'] = tokenizer.vocab_size
    model_config_dict['pad_token_id'] = tokenizer.pad_token_id

    model_config_obj = BaseLLMConfig(**model_config_dict)
    model = BaseLLM(model_config_obj)

    print(f"Model initialized with {get_total_params(model) / 1e9:.2f} Billion parameters.")

    # 8. Setup Optimizer and Scheduler
    print("Setting up optimizer and learning rate scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        betas=(float(config['adam_beta1']), float(config['adam_beta2'])),
        eps=float(config['adam_epsilon'])
    )
    num_training_steps = int(config['max_steps']) if int(config['max_steps']) > 0 else len(train_dataloader) // int(
        config['gradient_accumulation_steps']) * int(config['num_train_epochs'])
    lr_scheduler = get_scheduler(
        name=config['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * float(config['warmup_ratio'])),
        num_training_steps=num_training_steps
    )
    print(f"Total training steps: {num_training_steps}")

    # 9. Prepare for Distributed Training with Accelerate
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print("Model, Optimizer, DataLoaders, and Scheduler prepared for training.")

    # 10. Training Loop
    print("\n--- Training Started ---")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    while completed_steps < num_training_steps:
        for step, batch in enumerate(train_dataloader):
            if completed_steps >= num_training_steps:
                break

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                loss = outputs['loss']

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if accelerator.is_main_process:
                    log_dict = {"train_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}
                    accelerator.log(log_dict, step=completed_steps)
                    progress_bar.set_description(f"Loss: {loss.item():.4f}")

                save_steps = int(config.get('save_steps', 0))
                if completed_steps > 0 and save_steps > 0 and completed_steps % save_steps == 0:
                    output_path = Path(config['output_dir']) / f"step_{completed_steps}"
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)

                    if accelerator.is_main_process:
                        # **最终解决方案**: 采用 `safe_serialization=False` 来保证在任何环境下都能成功保存。
                        # 这是解决顽固的环境/库 bug 的最直接有效的方法。
                        print(f"Saving checkpoint at step {completed_steps} using safe_serialization=False...")
                        unwrapped_model.save_pretrained(str(output_path), safe_serialization=False)
                        tokenizer.save_pretrained(str(output_path))
                        print(f"Checkpoint saved at step {completed_steps} to {output_path}")

    progress_bar.close()
    print("--- Training Finished ---")

    if accelerator.is_main_process:
        final_model_path = Path(config['output_dir']) / "final_model"
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        # 同样，在最终保存时也使用 `safe_serialization=False`
        print(f"Saving final model to {final_model_path} using safe_serialization=False...")
        unwrapped_model.save_pretrained(str(final_model_path), safe_serialization=False)
        tokenizer.save_pretrained(str(final_model_path))
        print(f"Final model and tokenizer saved to {final_model_path}")
        if config.get('report_to') == "wandb":
            wandb.finish()

    print("\n--- [Bedrock] Pre-training Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/pretrain_trainer.py <path_to_config.yaml>")
        sys.exit(1)
    config_file_path = sys.argv[1]
    run_pretrain(config_file_path)