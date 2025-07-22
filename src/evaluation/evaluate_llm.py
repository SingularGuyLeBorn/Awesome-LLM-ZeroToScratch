# FILE: src/evaluation/evaluate_llm.py
"""
Bedrock Protocol: LLM Evaluation Script.

This module provides a functional tool for performing qualitative (subjective)
evaluation by generating responses to a set of predefined prompts. It also serves
as a guide for conducting quantitative (objective) evaluation using standard
benchmarks like `lm-evaluation-harness`.
"""

import sys
import yaml
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftConfig
import os
import shutil  # For cleaning up temporary directories
import gc  # For garbage collection


def load_model_for_evaluation(model_path: str):
    """Loads a PEFT adapter or a full model and prepares it for evaluation."""
    print(f"\n[Model Loader] Loading model from: {model_path}")

    # [FIX] Define an offload directory for CPU memory issues during evaluation
    # Create a unique temporary directory for this run's offloading
    evaluation_output_dir = Path("./checkpoints/evaluation_temp_models")
    offload_folder = evaluation_output_dir / "offload_eval_model"
    os.makedirs(offload_folder, exist_ok=True)
    print(f"--> No GPU detected. Configuring for CPU-only execution with offload to: {offload_folder}")

    peft_adapter_dir = Path(model_path)
    is_peft_adapter = (peft_adapter_dir / "adapter_config.json").exists()

    base_model_name_or_path = ""
    if is_peft_adapter:
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name_or_path = peft_config.base_model_name_or_path
            print(f"--> Detected PEFT adapter. Base model: {base_model_name_or_path}")
        except Exception as e:
            print(f"--> WARNING: Could not auto-detect base model from PEFT config ({e}).")
            # Fallback for our project structure
            base_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"--> Using fallback base model for our project: {base_model_name_or_path}")
    else:
        base_model_name_or_path = model_path
        print("--> Detected full model. Loading directly.")

    try:
        # Load the tokenizer first (no offloading needed for tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # [FIX] Load the base model with offloading
        print(f"--> Loading base model '{base_model_name_or_path}' with offloading...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map={"": "cpu"},
            trust_remote_code=True,
            attn_implementation="sdpa",
            offload_folder=str(offload_folder),  # Pass the offload directory
        )

        # If it's a PEFT adapter, load and merge it
        if is_peft_adapter:
            print("--> Applying and merging PEFT adapter (with offloading context)...")
            model = PeftModel.from_pretrained(model, model_path,
                                              offload_folder=str(offload_folder))  # Pass offload folder again
            model = model.merge_and_unload()
            print("--> LoRA weights merged into base model.")

        model.eval()
        print("[Model Loader] Model and tokenizer loaded successfully for evaluation.")
        return model, tokenizer, evaluation_output_dir  # Return the temp directory for cleanup
    except Exception as e:
        print(f"FATAL: Error loading model for evaluation: {e}")
        sys.exit(1)


def run_qualitative_evaluation(model, tokenizer):
    """Performs subjective evaluation by generating answers to predefined questions."""
    print("\n--- [Stage 1: Qualitative Evaluation (Manual Check)] ---")
    print("Generating responses for a set of predefined prompts...")

    prompts = [
        "What is the capital of France?",
        "Write a short, three-sentence horror story.",
        "Explain the concept of supervised fine-tuning in simple terms.",
        "Provide a python function to reverse a string.",
    ]

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for i, prompt_text in enumerate(prompts):
        print("\n" + "=" * 50)
        print(f"PROMPT {i + 1}/{len(prompts)}: {prompt_text}")
        print("=" * 50)

        chat = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)  # Move to model's device

        print(f"MODEL RESPONSE:")
        with torch.no_grad():
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=150,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
        print()  # for newline after streaming


def guide_quantitative_evaluation(model_path: str):
    """Provides guidance on how to run objective, benchmark-based evaluation."""
    print("\n--- [Stage 2: Quantitative Evaluation (Benchmark Guide)] ---")
    print("Quantitative evaluation measures model performance on standard academic benchmarks.")
    print("The industry-standard tool for this is `lm-evaluation-harness`.")
    print("\n**Why we don't run it directly here:**")
    print(
        " - Running benchmarks like MMLU or GSM8K is computationally intensive and can take hours or days, even on GPUs.")
    print(" - It requires a specific setup and large benchmark datasets to be downloaded.")

    print("\n**How to run it yourself (on a GPU machine):**")
    print("1. Install the tool:")
    print("   pip install lm-eval")
    print("\n2. Prepare your model:")
    print(
        "   - For our project, you first need a merged model. The qualitative evaluation above already uses a merged, in-memory model.")
    print("   - You would save this merged model to a new directory before running the harness.")

    print("\n3. Run the evaluation command:")
    print("   (This is a template, replace with your actual paths and desired tasks)")
    print(f"   lm_eval --model hf \\")
    print(f"       --model_args pretrained={model_path},dtype=float32 \\")
    print(f"       --tasks mmlu,gsm8k \\")
    print(f"       --device cpu \\")  # Keep CPU for general compatibility guide
    print(f"       --batch_size 1 \\")
    print(f"       --output_path ./evaluation_results.json")

    print("\nThis conceptual script has successfully loaded the model, which is the prerequisite for evaluation.")
    print("Please follow the steps above in a suitable environment to get official benchmark scores.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM evaluation.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint or PEFT adapter.")
    args = parser.parse_args()

    # Main execution flow
    model_for_eval, tokenizer_for_eval, temp_dir = load_model_for_evaluation(args.model_path)
    run_qualitative_evaluation(model_for_eval, tokenizer_for_eval)
    guide_quantitative_evaluation(args.model_path)

    # [FIX] Clean up temporary offload directory
    if temp_dir.exists():
        print(f"\n[Cleanup] Cleaning up temporary evaluation directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        gc.collect()

    print("\n--- [Bedrock] Evaluation Script Finished ---")

# END OF FILE: src/evaluation/evaluate_llm.py