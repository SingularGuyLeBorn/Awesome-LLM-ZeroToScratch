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
    The TRL DPOTrainer expects 'prompt', 'chosen', and 'rejected' columns.
    """
    # Assuming the dataset already has 'prompt', 'chosen', 'rejected' fields.
    # If your dataset format is different (e.g., 'question', 'answer_good', 'answer_bad'),
    # you would map them here.
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

    # Subset for quick demo if specified in config
    if 'dataset_subset_size' in config and config['dataset_subset_size'] > 0:
        dataset = dataset.select(range(config['dataset_subset_size']))

    dataset = dataset.map(format_dpo_dataset)
    print(f"Dataset loaded and formatted with {len(dataset)} samples.")

    # 3. Load SFT-tuned Model and Tokenizer
    print(f"Loading base SFT model for DPO: {config['model_name_or_path']}")
    # Determine the compute dtype for quantization based on config and GPU capabilities
    torch_dtype = torch.float16
    if config['bf16'] and (torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8):
        torch_dtype = torch.bfloat16
    elif config['fp16'] and torch.cuda.is_available():
        torch_dtype = torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8 else "sdpa",
    )
    model.config.use_cache = False  # Disable cache during training

    # The reference model for DPO is a non-trainable copy of the initial model.
    # TRL handles its creation automatically if you don't provide one, using the same config.

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    print("Model and tokenizer loaded successfully.")

    # 4. Configure PEFT (LoRA)
    print("Configuring PEFT with LoRA...")
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Configure Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim=config['optim'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],  # Added for consistency
        bf16=config['bf16'],
        fp16=config['fp16'],  # Added for consistency
        max_grad_norm=config['max_grad_norm'],  # Added for consistency
        logging_steps=config['logging_steps'],
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to=config['report_to'],
        run_name=config['run_name'],
        remove_unused_columns=False,  # Necessary for DPO with custom dataset columns
    )

    # 6. Initialize and Run DPO Trainer
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,  # TRL will handle creating the reference model from the base model
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=config['beta'],
        max_prompt_length=config['max_prompt_length'],
        max_length=config['max_length'],
        # eval_dataset=dataset_eval # Can add evaluation set if available
    )

    print("--- DPO Training Started ---")
    dpo_trainer.train()
    print("--- DPO Training Finished ---")

    # 7. Save Final Adapter Model and Tokenizer
    # Mandate of Empirical Proof: Save the model in a reproducible, standard format.
    final_model_path = Path(config['output_dir']) / "final_model"
    print(f"Saving final DPO-tuned adapter model to {final_model_path}...")
    dpo_trainer.save_model(str(final_model_path))  # Saves PEFT adapter and its config
    # Save the tokenizer separately to ensure it's available with the adapter
    tokenizer.save_pretrained(str(final_model_path))
    print("Model and tokenizer saved successfully.")

    print("\n--- [Bedrock] DPO Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/dpo_trainer.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv
    run_dpo(config_file_path)

# END OF FILE: src/trainers/dpo_trainer.py