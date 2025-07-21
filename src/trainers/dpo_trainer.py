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
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, PeftConfig
from trl import DPOTrainer


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
    """
    Main function to execute the DPO process.
    """
    print("--- [Bedrock] Initiating Direct Preference Optimization (DPO) ---")

    # 1. Load Configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Tokenizer FIRST
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # 3. Load and Format Dataset
    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        dataset = dataset.select(range(int(config['dataset_subset_size'])))

    formatting_function = format_dpo_dataset_factory(tokenizer)
    dataset = dataset.map(formatting_function)
    print(f"Dataset loaded and formatted with {len(dataset)} samples.")

    # [ULTIMATE FIX V7.0 - Force load into RAM]
    # This is our last resort for the stubborn meta tensor issue on CPU.
    # We will load the model fully into RAM, which relies on having enough virtual memory.

    print("Loading PEFT model and merging LoRA layers to force loading into RAM...")
    peft_config_for_base = PeftConfig.from_pretrained(config['model_name_or_path'])
    base_model_name = peft_config_for_base.base_model_name_or_path

    # Step 1: Load the base model fully onto CPU. No device_map, no offload.
    print(f"Loading base model '{base_model_name}' fully into RAM...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )

    # Step 2: Load the PEFT model on top of the base model
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, config['model_name_or_path'])

    # Step 3: Merge the LoRA layers into the base model
    print("Merging LoRA layers...")
    model = model.merge_and_unload()
    print("POLICY model prepared and fully loaded into RAM.")

    # We will not use a separate ref_model. DPOTrainer can create one from the merged model.
    # This minimizes the memory footprint to just one full model + its copy.
    ref_model = None

    # Garbage collect to free up memory from intermediate loading steps
    gc.collect()

    # 5. Configure PEFT (LoRA) for DPO training
    print("Configuring PEFT with LoRA for DPO training...")
    peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Configure Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
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
    )

    # 7. Initialize and Run DPO Trainer
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # Let TRL handle creation of the reference model from the merged model
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,  # Re-apply LoRA to the merged model for training
        beta=float(config['beta']),
        max_prompt_length=int(config['max_prompt_length']),
        max_length=int(config['max_length']),
    )

    print("--- DPO Training Started ---")
    dpo_trainer.train()
    print("--- DPO Training Finished ---")

    # 8. Save Final Adapter Model and Tokenizer
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