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
    dataset = load_dataset(config['dataset_name'], split="train")

    # [MODIFIED FOR CPU SPEED] Add a config option to subset the dataset for faster runs
    if 'dataset_subset_size_cpu' in config and int(config['dataset_subset_size_cpu']) > 0:
        subset_size = int(config['dataset_subset_size_cpu'])
        print(f"CPU mode: Subsetting dataset to first {subset_size} samples for speed.")
        dataset = dataset.select(range(subset_size))

    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. Load Model and Tokenizer
    print(f"Loading base model: {config['model_name_or_path']}")

    device_map = {"": "cpu"}
    quant_config = None
    torch_dtype = torch.float32
    attn_implementation = "sdpa"

    if torch.cuda.is_available():
        print("GPU detected. Preparing for GPU execution.")
        device_map = "auto"
        use_quantization = config.get('optim') == 'paged_adamw_8bit'

        if use_quantization:
            print("Applying 4-bit quantization for GPU.")
            compute_dtype = torch.float16
            if config.get('bf16', False) and torch.cuda.get_device_capability()[0] >= 8:
                compute_dtype = torch.bfloat16

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = compute_dtype

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            attn_implementation = "flash_attention_2"
            print("Using Flash Attention 2 for compatible GPU.")
    else:
        print("No GPU detected. Configuring for CPU-only execution.")

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        quantization_config=quant_config,
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

    print("Model and tokenizer loaded successfully.")

    # 4. Configure PEFT (LoRA)
    print("Configuring PEFT with LoRA...")
    peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("PEFT configured.")

    # 5. Configure Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=int(config['num_train_epochs']),
        per_device_train_batch_size=int(config['per_device_train_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        optim=config['optim'],
        save_steps=int(config['save_steps']),
        logging_steps=int(config['logging_steps']),
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        fp16=config.get('fp16', False) and torch.cuda.is_available(),
        bf16=config.get('bf16', False) and torch.cuda.is_available(),
        max_grad_norm=float(config['max_grad_norm']),
        max_steps=int(config['max_steps']),
        warmup_ratio=float(config['warmup_ratio']),
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to=config['report_to'],
        run_name=config['run_name'],
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
        max_seq_length=int(config['max_seq_length']),
        args=training_args,
        packing=True,
    )

    print("--- Training Started ---")
    trainer.train()
    print("--- Training Finished ---")

    # 7. Save Final Adapter Model and Tokenizer
    final_model_path = Path(config['output_dir']) / "final_model"
    print(f"Saving final adapter model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
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