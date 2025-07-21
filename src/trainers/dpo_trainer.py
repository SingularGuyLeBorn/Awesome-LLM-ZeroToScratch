# FILE: src/trainers/dpo_trainer.py
"""
Bedrock Protocol: Direct Preference Optimization (DPO) Trainer.

This script uses the Hugging Face TRL library to perform DPO, a form of
reinforcement learning from human feedback (RLHF) that is more stable and
computationally efficient than traditional PPO.
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
from trl import DPOTrainer


def format_dpo_dataset(example: dict) -> dict:
    """
    Formats a single example from the source dataset to the required DPO format.
    """
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def run_dpo(config_path: str) -> None:
    """
    Main function to execute the DPO process.

    Args:
        config_path: Path to the YAML configuration file.
    """
    print("--- [Bedrock] Initiating Direct Preference Optimization (DPO) ---")

    # 1. Load Configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Dataset
    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        dataset = dataset.select(range(int(config['dataset_subset_size'])))

    dataset = dataset.map(format_dpo_dataset)
    print(f"Dataset loaded and formatted with {len(dataset)} samples.")

    # 3. Load SFT-tuned Model and Tokenizer
    print(f"Loading base SFT model for DPO: {config['model_name_or_path']}")

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

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    # 5. Configure Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=int(config['num_train_epochs']),
        per_device_train_batch_size=int(config['per_device_train_batch_size']),
        per_device_eval_batch_size=int(config['per_device_eval_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        optim=config['optim'],
        # [FIXED] Ensure all numeric values are cast to their correct type
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config.get('weight_decay', 0.0)),
        bf16=config.get('bf16', False) and torch.cuda.is_available(),
        fp16=config.get('fp16', False) and torch.cuda.is_available(),
        max_grad_norm=float(config['max_grad_norm']),
        logging_steps=int(config['logging_steps']),
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to=config['report_to'],
        run_name=config['run_name'],
        remove_unused_columns=False,
    )

    # 6. Initialize and Run DPO Trainer
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=float(config['beta']),
        max_prompt_length=int(config['max_prompt_length']),
        max_length=int(config['max_length']),
    )

    print("--- DPO Training Started ---")
    dpo_trainer.train()
    print("--- DPO Training Finished ---")

    # 7. Save Final Adapter Model and Tokenizer
    final_model_path = Path(config['output_dir']) / "final_model"
    print(f"Saving final DPO-tuned adapter model to {final_model_path}...")
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