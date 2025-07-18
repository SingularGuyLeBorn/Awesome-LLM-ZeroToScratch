# FILE: data_processing/build_tokenizer.py
"""
Bedrock Protocol: Module for training a SentencePiece tokenizer. (Upgraded for Robustness)

This script provides a single, focused function to train a tokenizer from a
corpus file. It now includes a dynamic fallback mechanism to automatically
adjust the vocabulary size if the initial request is too large for the corpus,
ensuring the pipeline doesn't crash on smaller datasets.
"""

import sentencepiece as spm
from pathlib import Path
import argparse
import re # Import the regular expression module
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
        output_path_prefix: str,
        corpus_path: str,
        vocab_size: int,
        model_type: str,
        character_coverage: float,
        add_special_tokens: bool = True
) -> None:
    """
    Trains a SentencePiece tokenizer and saves the model.
    Includes a fallback to automatically reduce vocab_size if it's too high.
    """
    # Ensure the output directory exists
    output_dir = Path(output_path_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initial Training Attempt ---
    try:
        print(f"--- [Bedrock] Initiating SentencePiece Training (Attempt 1) ---")
        print(f"Requested Vocab Size: {vocab_size}")
        _run_sentencepiece_training(output_path_prefix, corpus_path, vocab_size, model_type, character_coverage, add_special_tokens)

    except RuntimeError as e:
        print(f"\nWarning: Initial training failed. This is common with small datasets.")
        print(f"Error details: {e}")

        # --- Dynamic Vocab Size Adjustment Logic ---
        # Use regular expression to find the suggested vocab size in the error message
        match = re.search(r'Please set it to a value <= (\d+)', str(e))
        if match:
            suggested_vocab_size = int(match.group(1))
            print(f"--- [Bedrock] Retrying with suggested Vocab Size: {suggested_vocab_size} ---")

            # --- Second Training Attempt ---
            try:
                _run_sentencepiece_training(output_path_prefix, corpus_path, suggested_vocab_size, model_type, character_coverage, add_special_tokens)
            except RuntimeError as e2:
                print(f"\nFatal Error: Retrying with suggested vocab size also failed.")
                print(f"Error details: {e2}")
                return # Exit the function on second failure
        else:
            print("\nFatal Error: Could not parse suggested vocab size from the error message. Aborting.")
            return # Exit if we can't parse the error

    print("\n--- [Bedrock] SentencePiece Training Complete ---")

    # --- Save in Hugging Face Transformers format ---
    print(f"--- [Bedrock] Saving Tokenizer in Hugging Face format... ---")
    hf_tokenizer_output_dir = output_dir / f"{Path(output_path_prefix).name}_hf"
    hf_tokenizer_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_for_hf = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(output_path_prefix).with_suffix('.model')),
        bos_token='[BOS]' if add_special_tokens else None,
        eos_token='[EOS]' if add_special_tokens else None,
        unk_token='[UNK]' if add_special_tokens else None,
        pad_token='[PAD]' if add_special_tokens else None,
    )
    if tokenizer_for_hf.pad_token is None and tokenizer_for_hf.eos_token is not None:
        tokenizer_for_hf.pad_token = tokenizer_for_hf.eos_token

    tokenizer_for_hf.save_pretrained(str(hf_tokenizer_output_dir))
    print(f"Hugging Face format tokenizer saved successfully to: {hf_tokenizer_output_dir}")


def _run_sentencepiece_training(
    output_path_prefix: str,
    corpus_path: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    add_special_tokens: bool
):
    """Helper function to construct and run the SentencePiece training command."""
    cmd_parts = [
        f'--input={corpus_path}',
        f'--model_prefix={output_path_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={model_type}',
        f'--character_coverage={character_coverage}',
    ]

    if add_special_tokens:
        cmd_parts.extend([
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3',
            f'--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS]',
            f'--user_defined_symbols=<0x0A>'
        ])

    command = " ".join(cmd_parts)
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer with dynamic vocab size fallback.")
    parser.add_argument(
        "--output_path_prefix", type=str, required=True,
        help="The base path and prefix for the output files (e.g., './data/tokenizers/my_spm')."
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True,
        help="Path to the text file corpus to train on."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8000,
        help="The desired total number of tokens in the vocabulary."
    )
    # ... (the rest of the __main__ block remains the same)
    parser.add_argument(
        "--model_type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"],
        help="The tokenizer model type ('unigram' is recommended)."
    )
    parser.add_argument(
        "--character_coverage", type=float, default=1.0,
        help="The percentage of characters in the corpus to be covered."
    )
    parser.add_argument(
        "--add_special_tokens", type=bool, default=True,
        help="Whether to add default special tokens."
    )
    args = parser.parse_args()

    train_tokenizer(
        output_path_prefix=args.output_path_prefix,
        corpus_path=args.corpus_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        add_special_tokens=args.add_special_tokens
    )