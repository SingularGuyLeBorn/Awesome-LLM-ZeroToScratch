# FILE: data_processing/download_and_reproduce.py
"""
基石协议：数据管道执行主脚本。

此脚本作为下载、处理和准备教程所需数据集的入口点。
它通过使用版本化数据集和确定性处理步骤，确保数据管道的完全可复现性。

设计为运行一次以设置所有必要的数据工件。
"""

import sys
import os
from pathlib import Path
import yaml
from datasets import load_dataset, concatenate_datasets, Dataset  # 导入 Dataset 类
import logging
import datetime
import shutil  # 导入 shutil 模块用于文件操作
# import json  # 导入 json 模块用于解析 COCO json (原先用于手动处理，现在已由 datasets 库处理，故不再直接使用)
from PIL import Image  # 导入 Image 用于图像加载

# from huggingface_hub import snapshot_download  # 导入 snapshot_download (对于数据集现在推荐直接使用 load_dataset，故不再直接使用)

# 配置日志
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f"data_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,  # 默认日志级别为 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # 输出到文件
        logging.StreamHandler(sys.stdout)  # 输出到控制台
    ]
)

# 获取主脚本的日志器
logger = logging.getLogger(__name__)

# 从 data_processing 包导入相关函数
from data_processing.process_text import clean_text_dataset, deduplicate_dataset, augment_text_dataset
from data_processing.process_vlm import process_vlm_dataset, deduplicate_vlm_dataset, augment_vlm_dataset
from data_processing.build_tokenizer import train_tokenizer


def run_text_pipeline(config_path: str) -> None:
    """
    执行用于文本预训练的整个数据管道。
    """
    logger.info("--- [基石] 启动文本数据处理流水线 ---")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"已加载配置文件: {config_path}")

    # 解析路径
    base_output_dir = Path(config['base_output_dir'])
    # 对于文本数据集，通常 dataset_name 就是 raw 和 processed 目录的一部分
    raw_data_dir = base_output_dir / 'raw' / config['dataset_name']
    processed_data_dir = base_output_dir / 'processed' / config['dataset_name']
    tokenizer_corpus_path = processed_data_dir / "corpus.txt"
    tokenizer_output_prefix = base_output_dir / 'tokenizers' / config['dataset_name'] / Path(
        config['dataset_name']).name

    # 确保输出目录存在
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"数据输出目录已准备: {processed_data_dir}")
    logger.info(f"分词器输出前缀已准备: {tokenizer_output_prefix}")

    logger.info(f"1. 正在加载数据集 '{config['dataset_name']}'...")
    # 从配置文件获取数据集子集大小。如果为0或未指定，则加载整个数据集。
    subset_size = config.get('dataset_subset_size')

    if subset_size is not None and subset_size > 0:
        logger.info(f"--- [基石] 正在加载数据集的子集: 前 {subset_size} 个样本。 ---")
        dataset = load_dataset(
            config['dataset_name'],
            config['dataset_config_name'],
            split={
                'train': f'train[:{subset_size}]',
                'validation': f'validation[:{int(subset_size * 0.1)}]',
                'test': f'test[:{int(subset_size * 0.1)}]'
            },
            cache_dir=str(raw_data_dir),  # datasets 库的 cache_dir 参数需要字符串
        )
    else:
        logger.info(f"--- [基石] 正在加载完整数据集。 ---")
        dataset = load_dataset(
            config['dataset_name'],
            config['dataset_config_name'],
            cache_dir=str(raw_data_dir),
        )
    logger.info("数据集加载成功。")

    logger.info("2. 正在进行文本清洗和质量过滤...")
    cleaned_dataset = clean_text_dataset(dataset, text_column=config['text_column'])
    logger.info("文本清洗和质量过滤完成。")

    logger.info("3. 正在进行数据去重...")
    deduplicated_dataset = deduplicate_dataset(cleaned_dataset, text_column=config['text_column'])
    logger.info("数据去重完成。")

    logger.info("4. 正在进行数据增强...")
    augmented_dataset = augment_text_dataset(deduplicated_dataset, text_column=config['text_column'])
    logger.info("数据增强步骤完成。")

    logger.info(f"5. 正在保存处理后的数据集到 '{processed_data_dir}'...")
    augmented_dataset.save_to_disk(str(processed_data_dir))
    logger.info(f"处理后的数据集已保存到: {processed_data_dir}")

    logger.info(f"6. 正在准备分词器训练语料到 '{tokenizer_corpus_path}'...")
    # 假设文本数据管道加载的数据集包含 'train', 'validation', 'test' 分割
    # 确保这些分割存在，或者根据实际情况调整
    full_text_dataset = concatenate_datasets([
        augmented_dataset['train'],
        augmented_dataset['validation'],
        augmented_dataset['test']
    ])
    with open(tokenizer_corpus_path, "w", encoding="utf-8") as f:
        for example in full_text_dataset:
            text = example[config['text_column']]
            if text:  # 确保写入非空文本
                f.write(text + "\n")
    logger.info("分词器训练语料已准备。")

    logger.info(f"7. 正在训练 SentencePiece 分词器并保存为 Hugging Face 格式...")
    train_tokenizer(
        output_path_prefix=str(tokenizer_output_prefix),  # 确保传递字符串
        corpus_path=str(tokenizer_corpus_path),  # 确保传递字符串
        vocab_size=config['vocab_size'],
        model_type=config['model_type'],
        character_coverage=config['character_coverage']
    )
    logger.info(f"分词器训练和保存完成到 '{tokenizer_output_prefix}_hf'.")

    logger.info("--- [基石] 文本数据处理流水线全部完成 ---")


def run_vlm_pipeline(config_path: str) -> None:
    """
    执行用于 VLM 预训练的数据管道。
    已重构，以更稳健地处理 COCO 数据下载和加载。
    """
    logger.info("\n--- [基石] 启动 VLM 数据处理流水线 ---")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"已加载配置文件: {config_path}")

    base_output_dir = Path(config['base_output_dir'])
    # 对于 VLM 数据，通常目录名为 'coco_demo' 或 'coco_val_demo'
    # 使用 Path(config['raw_data_dir']).name 来从配置文件中提取目录名部分
    raw_data_dir = base_output_dir / 'raw' / Path(config['raw_data_dir']).name
    processed_data_dir = base_output_dir / 'processed' / Path(config['processed_data_dir']).name

    # 确保输出目录存在
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"VLM 数据输出目录已准备: {processed_data_dir}")

    # --- VLM 数据下载和初始加载逻辑 ---
    dataset_name = config['dataset_name']
    # 从配置文件获取 num_samples_to_process。如果为0或未指定，则加载整个数据集。
    num_samples_to_process = config.get("num_samples_to_process")

    logger.info(f"1. 正在加载数据集 '{dataset_name}'...")

    # 直接使用 datasets.load_dataset 从 Hugging Face Hub 加载 VLM 数据
    if num_samples_to_process is not None and num_samples_to_process > 0:
        logger.info(f"--- [基石] 正在加载数据集的子集: 前 {num_samples_to_process} 个样本。 ---")
        # lmms-lab/COCO-Caption2017 数据集目前只有一个 'train' 分割
        dataset = load_dataset(
            dataset_name,
            split=f'train[:{num_samples_to_process}]',
            cache_dir=str(raw_data_dir),  # datasets 库的 cache_dir 参数需要字符串
        )
    else:
        logger.info(f"--- [基石] 正在加载完整数据集。 ---")
        # lmms-lab/COCO-Caption2017 数据集目前只有一个 'train' 分割，所以直接指定 split='train'
        dataset = load_dataset(
            dataset_name,
            split='train',
            cache_dir=str(raw_data_dir),
        )
    logger.info("数据集加载成功。")
    # --- VLM 数据下载和初始加载逻辑结束 ---

    logger.info("2. 正在处理 VLM 数据 (图像转换, 文本清洗等)...")
    processed_dataset = process_vlm_dataset(
        dataset,
        image_column=config['image_column'],
        text_column=config['text_column'],
        distillation_config=config['distillation']
    )
    logger.info("VLM 数据处理完成。")

    logger.info("3. 正在进行 VLM 数据去重...")
    deduplicated_dataset = deduplicate_vlm_dataset(
        processed_dataset,
        image_column=config['image_column'],  # 原始列名，仅为函数签名兼容性
        text_column='cleaned_captions'  # 确保使用清洗后的字幕进行去重
    )
    logger.info("VLM 数据去重完成。")

    logger.info("4. 正在进行 VLM 数据增强...")
    augmented_vlm_dataset = augment_vlm_dataset(
        deduplicated_dataset,
        image_column=config['image_column'],
        text_column='cleaned_captions',
        augmentation_config=config.get('augmentation', {})
    )
    logger.info("VLM 数据增强步骤完成。")

    logger.info(f"5. 正在保存处理后的数据集到 '{processed_data_dir}'...")
    augmented_vlm_dataset.save_to_disk(str(processed_data_dir))
    logger.info(f"处理后的 VLM 数据集已保存到: {processed_data_dir}")

    logger.info("--- [基石] VLM 数据处理流水线全部完成 ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行 Awesome-LLM-ZeroToScratch 的数据处理管道。")
    parser.add_argument(
        "pipeline",
        type=str,
        choices=["text", "vlm", "all"],
        help="要运行的管道: 'text', 'vlm', 或 'all'。"
    )
    args = parser.parse_args()

    PIPELINE_TO_RUN = args.pipeline

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    text_config = project_root / "configs/data/text_pretrain.yaml"
    vlm_config = project_root / "configs/data/vlm_pretrain.yaml"

    if PIPELINE_TO_RUN in ["text", "all"]:
        run_text_pipeline(str(text_config))

    if PIPELINE_TO_RUN in ["vlm", "all"]:
        run_vlm_pipeline(str(vlm_config))

    logger.info("\n[基石] 所有选定的流水线已成功执行。")

# END OF FILE: data_processing/download_and_reproduce.py