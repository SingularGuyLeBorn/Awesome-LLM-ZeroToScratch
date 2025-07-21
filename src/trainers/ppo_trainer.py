# FILE: src/trainers/ppo_trainer.py
"""
Bedrock Protocol: Proximal Policy Optimization (PPO) Trainer (Conceptual).

This script provides a conceptual outline for using the Hugging Face TRL library
to perform PPO, common reinforcement learning from human feedback (RLHF) algorithm.
"""
import sys
from pathlib import Path
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# Ensure project root is in path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.append(str(project_root))


def run_ppo(config_path: str) -> None:
    """
    Conceptual main function to execute the PPO process.

    Args:
        config_path: Path to the YAML configuration file.
    """
    print("--- [Bedrock] Initiating Proximal Policy Optimization (PPO) (Conceptual) ---")

    # 1. Load Configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Dataset
    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'], split="train")

    if 'dataset_subset_size' in config and int(config['dataset_subset_size']) > 0:
        dataset = dataset.select(range(int(config['dataset_subset_size'])))

    def format_ppo_dataset(example: dict) -> dict:
        return {"query": example[config['dataset_text_field']]}

    dataset = dataset.map(format_ppo_dataset, remove_columns=dataset.column_names)
    print(f"Dataset loaded and formatted with {len(dataset)} prompts.")

    # 3. Load SFT-tuned Model and Tokenizer (Actor Model)
    print(f"Loading SFT model as Actor for PPO: {config['model_name_or_path']}")

    device_map = {"": "cpu"}
    quant_config = None
    torch_dtype = torch.float32
    attn_implementation = "sdpa"

    if torch.cuda.is_available():
        print("GPU detected. Preparing for GPU execution.")
        device_map = "auto"
        use_quantization = config.get('bf16', False) or config.get('fp16', False)

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
    else:
        print("No GPU detected. Configuring for CPU-only execution.")

    lora_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config['model_name_or_path'],
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        peft_config=lora_config
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Actor Model and tokenizer loaded successfully.")

    # 5. Configure PPO Trainer Arguments
    print("Setting up PPO training arguments...")
    ppo_config = PPOConfig(
        exp_name=config['run_name'],
        log_with=config.get('report_to'),
        model_name=config['model_name_or_path'],
        # [FIXED] Ensure all numeric values are cast to their correct type
        steps=int(config['ppo_steps']),
        learning_rate=float(config['learning_rate']),
        batch_size=int(config['batch_size']),
        mini_batch_size=int(config['ppo_mini_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        ppo_epochs=int(config['ppo_epochs']),
        init_kl_coef=float(config['init_kl_coef']),
        target_kl=float(config.get('target_kl', 0.1)),
        adap_kl_ctrl=bool(config['adap_kl_ctrl']),
        seed=int(config['seed']),
        remove_unused_columns=False
    )

    # 6. Initialize Reward Model (Conceptual)
    print("Initializing conceptual Reward Model...")

    def get_dummy_reward(outputs):
        return [torch.tensor(float(len(o))) for o in outputs]

    # 7. Initialize and Run PPOTrainer
    print("Initializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("--- PPO Training Started (Conceptual) ---")
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": int(config['max_new_tokens']),
    }

    for epoch in range(int(config['ppo_epochs_conceptual'])):
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            if batch_idx >= int(config['ppo_num_batches_conceptual']):
                break

            query_tensors = batch['input_ids']
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = tokenizer.batch_decode(response_tensors)

            rewards = get_dummy_reward(batch["response"])

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            print(
                f"Conceptual PPO Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {stats['ppo/loss/total']:.4f}, Reward = {stats['ppo/rewards/mean']:.4f}")

    print("--- PPO Training Finished (Conceptual) ---")

    # 8. Save Final Model
    final_model_path = Path(config['output_dir']) / "final_ppo_model"
    print(f"Saving final PPO-tuned adapter model to {final_model_path}...")
    ppo_trainer.save_model(str(final_model_path))
    # Note: tokenizer is already part of the model saved by PPOTrainer
    print("Model and tokenizer saved successfully.")

    print("\n--- [Bedrock] PPO Process Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/trainers/ppo_trainer.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    run_ppo(config_file_path)

# END OF FILE: src/trainers/ppo_trainer.py