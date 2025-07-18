# FILE: data_processing/download_and_reproduce.py
"""
Bedrock Protocol: Main script for data pipeline execution.

This script serves as the entry point for downloading, processing, and preparing
the datasets required for the tutorial. It ensures full reproducibility of the
data pipeline by using versioned datasets and deterministic processing steps.

It's designed to be run once to set up all necessary data artifacts.
"""

import sys
import os
from pathlib import Path
import yaml
from datasets import load_dataset, concatenate_datasets

# Ensure the script can find the 'src' and other modules
sys.path.append(str(Path(__file__).resolve().parents))

from data_processing.process_text import clean_text_dataset
from data_processing.process_vlm import process_vlm_dataset
from data_processing.build_tokenizer import train_tokenizer


def run_text_pipeline(config_path: str) -> None:
    """
    Executes the entire data pipeline for text pre-training.
    """
    print("--- [Bedrock] Initiating Text Data Pipeline ---")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve paths
    base_output_dir = Path(config['base_output_dir'])
    raw_data_dir = base_output_dir / 'raw' / config['dataset_name']
    processed_data_dir = base_output_dir / 'processed' / config['dataset_name']
    tokenizer_corpus_path = processed_data_dir / "corpus.txt"
    # Ensure tokenizer output path points to a directory for HF format
    tokenizer_output_prefix = base_output_dir / 'tokenizers' / config['dataset_name'] / Path(
        config['dataset_name']).name

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_output_prefix.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dir for tokenizer output

    print(f"1. Loading dataset '{config['dataset_name']}'...")
    # Using a subset for faster demonstration. For full pre-training, remove [:10000]
    # 检查配置中是否有子集大小的设置
    subset_size = config.get('dataset_subset_size')

    if subset_size and subset_size > 0:
        print(f"--- [Bedrock] Loading a subset of the dataset: first {subset_size} samples. ---")
        # 为每个split都加载一个子集
        dataset = load_dataset(
            config['dataset_name'],
            config['dataset_config_name'],
            split={
                'train': f'train[:{subset_size}]',
                'validation': f'validation[:{int(subset_size * 0.1)}]',  # 验证集取10%
                'test': f'test[:{int(subset_size * 0.1)}]'  # 测试集取10%
            },
            cache_dir=str(raw_data_dir),
        )
    else:
        print(f"--- [Bedrock] Loading the full dataset. ---")
        dataset = load_dataset(
            config['dataset_name'],
            config['dataset_config_name'],
            cache_dir=str(raw_data_dir),
        )
    print("Dataset loaded successfully.")

    print("2. Cleaning and processing text data...")
    cleaned_dataset = clean_text_dataset(dataset, text_column=config['text_column'])
    print("Text data cleaned.")

    print(f"3. Saving processed dataset to '{processed_data_dir}'...")
    cleaned_dataset.save_to_disk(str(processed_data_dir))
    print(f"Processed dataset saved to: {processed_data_dir}")

    print(f"4. Preparing corpus for tokenizer training at '{tokenizer_corpus_path}'...")
    # Concatenate all text from train, validation, and test splits for tokenizer training
    full_text_dataset = concatenate_datasets([
        cleaned_dataset['train'],
        cleaned_dataset['validation'],
        cleaned_dataset['test']
    ])
    with open(tokenizer_corpus_path, "w", encoding="utf-8") as f:
        for example in full_text_dataset:
            f.write(example[config['text_column']] + "\n")
    print("Corpus prepared.")

    print(f"5. Training SentencePiece tokenizer and saving in HF format...")
    # Call the build_tokenizer.py's train_tokenizer function
    train_tokenizer(
        output_path_prefix=str(tokenizer_output_prefix),
        corpus_path=str(tokenizer_corpus_path),
        vocab_size=config['vocab_size'],
        model_type=config['model_type'],
        character_coverage=config['character_coverage']
    )
    print(f"Tokenizer trained and saved in HF format to '{tokenizer_output_prefix}_hf'.")

    print("--- [Bedrock] Text Data Pipeline Complete ---")


def run_vlm_pipeline(config_path: str) -> None:
    """
    Executes the data pipeline for VLM pre-training.
    """
    print("\n--- [Bedrock] Initiating VLM Data Pipeline ---")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_output_dir = Path(config['base_output_dir'])
    raw_data_dir = base_output_dir / 'raw' / 'coco_demo'
    processed_data_dir = base_output_dir / 'processed' / 'coco_demo'

    processed_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"1. Loading a subset of '{config['dataset_name']}'...")
    # For VLM, we take a small, reproducible slice for demonstration purposes.
    dataset = load_dataset(
        config['dataset_name'],
        split=f'train[:{config["num_samples_to_process"]}]',
        cache_dir=str(raw_data_dir)
    )
    print(f"Loaded {len(dataset)} samples.")

    print("2. Processing VLM data (image handling, text cleaning)...")
    processed_dataset = process_vlm_dataset(
        dataset,
        image_column=config['image_column'],
        text_column=config['text_column'],
        distillation_config=config['distillation']
    )
    print("VLM data processed.")

    print(f"3. Saving processed dataset to '{processed_data_dir}'...")
    processed_dataset.save_to_disk(str(processed_data_dir))
    print(f"Processed VLM dataset saved to: {processed_data_dir}")

    print("--- [Bedrock] VLM Data Pipeline Complete ---")


if __name__ == "__main__":
    # 使用 argparse 进行更健壮的命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="Run the data processing pipeline for Awesome-LLM-ZeroToScratch.")
    parser.add_argument(
        "pipeline",
        type=str,
        choices=["text", "vlm", "all"],
        help="The pipeline to run: 'text', 'vlm', or 'all'."
    )
    args = parser.parse_args()

    PIPELINE_TO_RUN = args.pipeline

    # 定义配置文件的路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    text_config = project_root / "configs/data/text_pretrain.yaml"
    vlm_config = project_root / "configs/data/vlm_pretrain.yaml"

    if PIPELINE_TO_RUN in ["text", "all"]:
        run_text_pipeline(str(text_config))

    if PIPELINE_TO_RUN in ["vlm", "all"]:
        run_vlm_pipeline(str(vlm_config))

    print("\n[Bedrock] All selected pipelines have been successfully executed.")

# END OF FILE: data_processing/download_and_reproduce.py