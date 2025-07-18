# FILE: src/inference/inference.py
"""
Bedrock Protocol: Model Inference Script.

This script provides a simple command-line interface (CLI) for loading a
fine-tuned language model and interacting with it. It demonstrates how to
load a base model and its associated LoRA adapters.
"""

import torch
import sys
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def run_inference_cli(model_path: str, max_new_tokens: int = 200) -> None:
    """
    Loads a model (with optional LoRA adapter) and provides a CLI for interaction.

    Args:
        model_path: Path to the directory containing the base model (if no adapter)
                    or the LoRA adapter (if base model is from Hugging Face Hub).
                    For LoRA, this should be the path to the saved PEFT adapter,
                    e.g., './checkpoints/sft-tinyllama-guanaco/final_model'.
        max_new_tokens: Maximum number of tokens to generate per response.
    """
    print("--- [Bedrock] Starting Model Inference CLI ---")
    print(f"Attempting to load model from: {model_path}")

    # Determine if it's a PEFT adapter or a full model.
    # A PEFT adapter directory usually contains 'adapter_config.json'.
    peft_adapter_dir = Path(model_path)
    is_peft_adapter = (peft_adapter_dir / "adapter_config.json").exists()

    base_model_name_or_path = ""
    if is_peft_adapter:
        # If it's a PEFT adapter, we need to load the base model first.
        # The base model's original ID is typically stored in the adapter_config.json.
        try:
            # Mandate of Proactive Defense: Try to get base model from adapter config first.
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name_or_path = peft_config.base_model_name_or_path
            print(f"Detected PEFT adapter. Base model: {base_model_name_or_path}")
        except Exception as e:
            # Fallback for tutorial: If adapter_config doesn't contain base_model_name_or_path
            # or if it's a custom model from scratch.
            print(f"Warning: Could not extract base model from PEFT config ({e}).")
            print("Attempting to infer base model from SFT config for tutorial purposes.")
            try:
                # 修正: 更稳健地推断项目根目录和 SFT 配置路径
                script_path = Path(__file__).resolve()
                project_root = script_path.parent.parent.parent # Assuming script is in src/inference/
                sft_config_path = project_root / "configs/training/finetune_sft.yaml"
                with open(sft_config_path, 'r') as f:
                    sft_config = yaml.safe_load(f)
                base_model_name_or_path = sft_config['model_name_or_path']
                print(f"Inferred base model from SFT config: {base_model_name_or_path}")
            except Exception as e_sft:
                print(f"Error inferring base model from SFT config: {e_sft}")
                print("Using default fallback base model. Ensure this is correct for your adapter!")
                base_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fallback for demo
    else:
        # If it's not an adapter, it's assumed to be a full, standalone model.
        base_model_name_or_path = model_path
        print("Detected full model. Loading directly.")

    try:
        # Load the tokenizer
        # For PEFT adapters, tokenizer should be saved alongside the adapter or loaded from base model path.
        tokenizer_load_path = model_path if is_peft_adapter and (
                    peft_adapter_dir / "tokenizer.json").exists() else base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
        tokenizer.padding_side = "right"  # Llama models often prefer right padding during inference
        print("Tokenizer loaded.")

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8 else torch.float16,
            device_map="auto",  # Automatically map to GPU if available
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8 else "sdpa",
        )
        print("Base model loaded.")

        # Load PEFT adapter if applicable
        if is_peft_adapter:
            model = PeftModel.from_pretrained(model, model_path)
            print(f"PEFT adapter loaded from {model_path}.")
            # Mandate of Proactive Defense: Merge LoRA weights for faster inference and simpler deployment.
            model = model.merge_and_unload()
            print("LoRA weights merged into base model for inference.")

        model.eval()  # Set model to evaluation mode
        print("Model set to evaluation mode. Ready for inference.")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print(
            "Please ensure the model path is correct and all dependencies (transformers, peft, torch, bitsandbytes) are installed.")
        print("If loading a PEFT adapter, ensure the base model can be found or specified correctly.")
        sys.exit(1)

    print("\n--- Start Chatting (type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            # Format input for chat models (e.g., Llama-2-Chat, TinyLlama-Chat).
            # This uses tokenizer.apply_chat_template if available for standardized formatting.
            # If your model does not have a chat template, you might need custom formatting.
            messages = [{"role": "user", "content": user_input}]
            # add_generation_prompt=True ensures the template includes a token
            # to signal the model to start generating its response (e.g., assistant:).
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Enable sampling for more diverse outputs
                    top_p=0.9,  # Nucleus sampling parameter
                    temperature=0.7,  # Controls randomness. Lower is more deterministic.
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id  # Crucial for generation with padding
                )

            # Decode the generated tokens. Skip input tokens for cleaner output.
            # 修正: 确保切片索引正确，outputs 是一个张量
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Model: {generated_text.strip()}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            continue

# END OF FILE: src/inference/inference.py