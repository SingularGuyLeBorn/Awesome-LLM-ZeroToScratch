# FILE: src/trainers/sft_trainer.py
"""
Bedrock Protocol: Supervised Fine-Tuning (SFT) Trainer.

This script uses the Hugging Face TRL library to perform SFT on a language model
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. It is designed to be
driven by a YAML configuration file.
"""

import sys
from pathlib import Path
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer


def run_sft(config_path: str) -> None:
    """
    Main function to execute the SFT process.

    Args:
        config_path: Path to the YAML configuration file.
    """
    print("--- [Bedrock] Initiating Supervised Fine-Tuning (SFT) ---")

    # 1. Load Configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Dataset
    print(f"Loading dataset: {config['dataset_name']}")
    # For a smaller demo, consider using a subset: dataset = load_dataset(..., split="train[:1000]")
    dataset = load_dataset(config['dataset_name'], split="train")
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. Load Model and Tokenizer
    print(f"Loading base model: {config['model_name_or_path']}")

    # Mandate of Proactive Defense: Use quantization to reduce memory footprint.
    # Convert string 'bf16' to torch.bfloat16 (and 'fp16' to torch.float16)
    torch_dtype = torch.float16
    if config['bf16'] and (torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8):
        torch_dtype = torch.bfloat16
    elif config['fp16'] and torch.cuda.is_available():
        torch_dtype = torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch_dtype,  # Compute in bf16 or fp16
        bnb_4bit_use_double_quant=True,  # Apply double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        quantization_config=quant_config,
        device_map="auto",  # Automatically map model layers to available devices
        trust_remote_code=True,
        # Use FlashAttention 2 if available and compatible GPU (Ampere or newer)
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8 else "sdpa",
    )
    model.config.use_cache = False  # Disable cache for training speed
    model.config.pretraining_tp = 1  # Required for DeepSpeed zero3 sometimes

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name_or_path'],
        trust_remote_code=True
    )
    # Ensure padding token is set, crucial for batching and generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Llama models often prefer right padding during inference/training

    print("Model and tokenizer loaded successfully.")

    # 4. Configure PEFT (LoRA)
    print("Configuring PEFT with LoRA...")
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",  # Common setting for LoRA on bias weights
        task_type="CAUSAL_LM",  # Specify task type
    )
    print("PEFT configured.")

    # 5. Configure Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim=config['optim'],
        save_steps=config['save_steps'],
        logging_steps=config['logging_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        fp16=config['fp16'],
        bf16=config['bf16'],
        max_grad_norm=config['max_grad_norm'],
        max_steps=config['max_steps'],
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to=config['report_to'],
        run_name=config['run_name'],
        # Evaluation is optional, can be enabled with eval_steps, etc.
        # evaluation_strategy="steps",
        # eval_steps=config['eval_steps'],
        # save_total_limit=config['save_total_limit'] # Limit number of checkpoints
    )
    print("Training arguments set.")

    # 6. Initialize and Run Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=config['dataset_text_field'],
        max_seq_length=config['max_seq_length'],
        args=training_args,
        packing=False,  # Set to True for more efficient packing of short examples into longer sequences.
        # Can provide better throughput but might complicate dataset debugging.
    )

    print("--- Training Started ---")
    trainer.train()
    print("--- Training Finished ---")

    # 7. Save Final Adapter Model and Tokenizer
    # Mandate of Empirical Proof: Save the model in a reproducible, standard format.
    # The adapter is saved, not the full model, to keep file size small.
    final_model_path = Path(config['output_dir']) / "final_model"
    print(f"Saving final adapter model to {final_model_path}...")
    trainer.save_model(str(final_model_path))  # Saves PEFT adapter and its config
    # Save the tokenizer separately to ensure it's available with the adapter
    tokenizer.save_pretrained(str(final_model_path))
    print("Model and tokenizer saved successfully.")

    print("\n--- [Bedrock] SFT Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/sft_trainer.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv
    run_sft(config_file_path)

# END OF FILE: src/trainers/sft_trainer.py