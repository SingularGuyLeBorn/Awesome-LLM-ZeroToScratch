# FILE: src/inference/inference.py
"""
Bedrock Protocol: Model Inference Script.

This script provides a simple command-line interface (CLI) for loading a
fine-tuned or pre-trained language model and interacting with it.
Includes streaming for real-time feedback.
"""

import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Ensure our custom model is registered when this module is run
import src.models as models


def run_inference_cli(model_path: str, max_new_tokens: int = 20) -> None:  # <<< MODIFIED: Default tokens reduced to 20
    """
    Loads a model and provides a CLI for interaction with real-time streaming.

    Args:
        model_path: Path to the directory containing the model checkpoint.
        max_new_tokens: Maximum number of tokens to generate per response.
                        Default is low for fast CPU feedback.
    """
    print("--- [Bedrock] Starting Model Inference CLI ---")
    print(f"Attempting to load model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("Tokenizer loaded.")

        print(f"Loading model with AutoModelForCausalLM from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if 'cuda' in str(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")) and torch.cuda.get_device_capability()[
                                              0] >= 8 else torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        print("Model loaded successfully using AutoModel.")

        model.eval()
        print("Model set to evaluation mode. Ready for inference.")

    except Exception as e:
        print(f"Fatal error loading model or tokenizer: {e}")
        print("\nPlease ensure the model path is correct and all dependencies are installed.")
        sys.exit(1)

    # +++ START OF THE FIX: Add a TextStreamer +++
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # +++ END OF THE FIX +++

    print("\n--- Start Chatting (type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

            # Use a dictionary for generation arguments for clarity
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "streamer": streamer,  # <<< MODIFIED: Pass the streamer here
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.7,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id
            }

            print("Model: ", end="", flush=True)  # Print the prompt for the streamer

            with torch.no_grad():
                # .generate() will now print tokens as they are created
                model.generate(**generation_kwargs)

            # The streamer handles the printing, so we just need a newline
            print()

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM inference CLI.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint.")
    # MODIFIED: Changed default to a smaller, CPU-friendly value
    parser.add_argument("max_new_tokens", nargs='?', type=int, default=20,
                        help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    run_inference_cli(args.model_path, args.max_new_tokens)