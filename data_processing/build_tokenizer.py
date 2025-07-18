# FILE: data_processing/build_tokenizer.py
"""
Bedrock Protocol: Module for training a SentencePiece tokenizer.

This script provides a single, focused function to train a tokenizer from a
corpus file. It exposes key parameters and follows best practices for creating
a robust Unigram tokenizer. After training, it saves the tokenizer in both
SentencePiece native format and Hugging Face Transformers compatible format.
"""

import sentencepiece as spm
from pathlib import Path
import argparse
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
    Trains a SentencePiece tokenizer and saves the model and vocab files.
    Also saves it in a Hugging Face Transformers compatible format.

    Args:
        output_path_prefix: The base path and prefix for the output files
                            (e.g., './data/tokenizers/my_spm'). The script will
                            generate '.model' and '.vocab' files, and a directory
                            for Hugging Face format.
        corpus_path: Path to the text file corpus to train on.
        vocab_size: The total number of tokens in the vocabulary.
        model_type: The tokenizer model type (e.g., 'unigram', 'bpe').
                    'unigram' is highly recommended.
        character_coverage: The percentage of characters in the corpus to be
                            covered by the tokenizer. 1.0 is fine for most
                            alphabetic languages. For languages with many
                            characters (like CJK), a value like 0.9995 is
                            recommended to avoid including rare characters.
        add_special_tokens: If True, adds default special tokens ([PAD], [UNK], [BOS], [EOS]).
    """
    # Mandate of Intentionality: Every parameter is explicitly passed and justified.
    cmd_parts = [
        f'--input={corpus_path}',
        f'--model_prefix={output_path_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={model_type}',
        f'--character_coverage={character_coverage}',
    ]

    if add_special_tokens:
        # These are standard special tokens for most LLMs.
        # Note: If your model requires specific special tokens, ensure consistency here.
        cmd_parts.extend([
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3',
            f'--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS]',
            f'--user_defined_symbols=<0x0A>'  # Treat newline as a single token for better text generation
        ])

    command = " ".join(cmd_parts)

    print("--- [Bedrock] Initiating SentencePiece Training ---")
    print(f"Corpus: {corpus_path}")
    print(f"Output Prefix: {output_path_prefix}")
    print(f"Vocab Size: {vocab_size}")

    # Ensure the output directory exists
    output_dir = Path(output_path_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mandate of Empirical Proof: The training process is deterministic.
    spm.SentencePieceTrainer.Train(command)

    print("--- [Bedrock] SentencePiece Training Complete ---")
    print(f"SentencePiece model saved to: {output_path_prefix}.model")
    print(f"SentencePiece vocabulary saved to: {output_path_prefix}.vocab")

    # --- Save in Hugging Face Transformers format ---
    # This is crucial for easy loading with AutoTokenizer.from_pretrained()
    hf_tokenizer_output_dir = output_dir / f"{Path(output_path_prefix).name}_hf"
    hf_tokenizer_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- [Bedrock] Saving Tokenizer in Hugging Face format to: {hf_tokenizer_output_dir} ---")

    # Load the trained SentencePiece model and wrap it with PreTrainedTokenizerFast
    # This allows it to be used seamlessly with Hugging Face models.
    tokenizer_for_hf = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(output_path_prefix).with_suffix('.model')),
        # Fallback to defaults if special tokens are not explicitly in the SentencePiece model vocab
        bos_token='[BOS]' if add_special_tokens else None,
        eos_token='[EOS]' if add_special_tokens else None,
        unk_token='[UNK]' if add_special_tokens else None,
        pad_token='[PAD]' if add_special_tokens else None,
    )

    # Ensure pad token is set correctly for generation
    if tokenizer_for_hf.pad_token is None and tokenizer_for_hf.eos_token is not None:
        tokenizer_for_hf.pad_token = tokenizer_for_hf.eos_token

    tokenizer_for_hf.save_pretrained(str(hf_tokenizer_output_dir))
    print("Hugging Face format tokenizer saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument(
        "--output_path_prefix",
        type=str,
        required=True,
        help="The base path and prefix for the output files (e.g., './data/tokenizers/my_spm')."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to the text file corpus to train on."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="The total number of tokens in the vocabulary."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="The tokenizer model type ('unigram' is recommended)."
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=1.0,
        help="The percentage of characters in the corpus to be covered."
    )
    parser.add_argument(
        "--add_special_tokens",
        type=bool,
        default=True,
        help="Whether to add default special tokens ([PAD], [UNK], [BOS], [EOS])."
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

# END OF FILE: data_processing/build_tokenizer.py