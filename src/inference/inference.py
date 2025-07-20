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
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

# +++ START OF CRITICAL FIX FOR MODULE IMPORT +++
# 确保项目根目录在 Python 路径中，以便正确导入 src 模块
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent  # Assuming script is in src/inference/
sys.path.append(str(project_root))
# +++ END OF CRITICAL FIX FOR MODULE IMPORT +++

# 导入您的自定义模型类
from src.models.language_model import BaseLLM, BaseLLMConfig


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

    # 判断是否是 PEFT 适配器
    peft_adapter_dir = Path(model_path)
    is_peft_adapter = (peft_adapter_dir / "adapter_config.json").exists()

    # 判断是否是直接保存的 BaseLLM 检查点
    is_custom_basellm_checkpoint = False
    if not is_peft_adapter and (Path(model_path) / "config.json").exists():
        try:
            with open(Path(model_path) / "config.json", 'r', encoding='utf-8') as f:
                loaded_config_dict = json.load(f)
            if loaded_config_dict.get("model_type") == "BaseLLM":
                is_custom_basellm_checkpoint = True
        except Exception:
            pass  # 如果文件不存在或格式错误，则忽略

    base_model_name_or_path = ""
    if is_peft_adapter:
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name_or_path = peft_config.base_model_name_or_path
            print(f"Detected PEFT adapter. Base model: {base_model_name_or_path}")
        except Exception as e:
            print(f"Warning: Could not extract base model from PEFT config ({e}).")
            print("Attempting to infer base model from SFT config for tutorial purposes.")
            try:
                # 注意：这里的 project_root 已经由文件开头的代码处理了
                sft_config_path = project_root / "configs/training/finetune_sft.yaml"
                with open(sft_config_path, 'r', encoding='utf-8') as f:
                    sft_config = yaml.safe_load(f)
                base_model_name_or_path = sft_config['model_name_or_path']
                print(f"Inferred base model from SFT config: {base_model_name_or_path}")
            except Exception as e_sft:
                print(f"Error inferring base model from SFT config: {e_sft}")
                print("Using default fallback base model. Ensure this is correct for your adapter!")
                base_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif is_custom_basellm_checkpoint:
        base_model_name_or_path = model_path
        print(f"Detected direct custom BaseLLM checkpoint: {model_path}")
    else:
        base_model_name_or_path = model_path
        print("Detected full standard Hugging Face model. Loading directly.")

    try:
        tokenizer_load_path = model_path if (is_peft_adapter or is_custom_basellm_checkpoint) and (
                    Path(model_path) / "tokenizer.json").exists() else base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("Tokenizer loaded.")

        if is_custom_basellm_checkpoint:
            print(f"Manually loading custom BaseLLM from checkpoint: {base_model_name_or_path}")
            model_config = BaseLLMConfig.from_pretrained(base_model_name_or_path)
            model = BaseLLM(model_config)
            model_state_dict = torch.load(Path(base_model_name_or_path) / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(model_state_dict)
            print("Custom BaseLLM loaded and weights applied successfully.")
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            print("Loading model using AutoModelForCausalLM (for standard HF models or PEFT base).")

            quant_config = None
            if torch.cuda.is_available():
                if torch.cuda.get_device_capability() >= 8:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    quant_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8 else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability() >= 8 else "sdpa",
            )
            print("Model loaded.")

            if is_peft_adapter:
                model = PeftModel.from_pretrained(model, model_path)
                print(f"PEFT adapter loaded from {model_path}.")
                model = model.merge_and_unload()
                print("LoRA weights merged into base model for inference.")

        model.eval()
        print("Model set to evaluation mode. Ready for inference.")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure the model path is correct and all dependencies are installed.")
        print("\n=== DEBUG HINT ===")
        print(
            "If the error mentions 'BaseLLM' not recognized, it means the custom model class definition cannot be found.")
        print("Ensure 'src/models/language_model.py' is correct and accessible.")
        print("If you are on CPU, 'bitsandbytes' GPU warnings are normal and can be ignored.")
        sys.exit(1)

    print("\n--- Start Chatting (type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            messages = [{"role": "user", "content": user_input}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Model: {generated_text.strip()}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM inference CLI.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint or PEFT adapter.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    run_inference_cli(args.model_path, args.max_new_tokens)