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
from datasets import load_dataset, concatenate_datasets, Dataset  # 导入 Dataset 类
import logging
import datetime
import shutil  # 导入 shutil 模块用于文件操作
import json  # 导入 json 模块用于解析 COCO json
from PIL import Image  # 导入 Image 用于图像加载
from huggingface_hub import snapshot_download  # 导入 snapshot_download

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
    Executes the entire data pipeline for text pre-training.
    """
    logger.info("--- [Bedrock] 启动文本数据处理流水线 ---")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"已加载配置文件: {config_path}")

    # 解析路径
    base_output_dir = Path(config['base_output_dir'])
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
    subset_size = config.get('dataset_subset_size')

    if subset_size is not None and subset_size > 0:
        logger.info(f"--- [Bedrock] 正在加载数据集的子集: 前 {subset_size} 个样本。 ---")
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
        logger.info(f"--- [Bedrock] 正在加载完整数据集。 ---")
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

    logger.info("--- [Bedrock] 文本数据处理流水线全部完成 ---")


def run_vlm_pipeline(config_path: str) -> None:
    """
    Executes the data pipeline for VLM pre-training.
    Refactored to handle COCO data download and loading more robustly.
    """
    logger.info("\n--- [Bedrock] 启动 VLM 数据处理流水线 ---")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"已加载配置文件: {config_path}")

    base_output_dir = Path(config['base_output_dir'])
    raw_data_dir = base_output_dir / 'raw' / 'coco_demo'
    processed_data_dir = base_output_dir / 'processed' / 'coco_demo'

    # 确保输出目录存在
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"VLM 数据输出目录已准备: {processed_data_dir}")

    # --- 重构的 VLM 数据下载和初始加载逻辑 ---
    dataset_name = config['dataset_name']  # "HuggingFaceM4/COCO"
    num_samples_to_process = config["num_samples_to_process"]

    logger.info(f"1. 正在显式下载数据集 '{dataset_name}' 的原始文件到本地缓存目录 '{raw_data_dir}'...")

    # 先清理旧的缓存，以防下载损坏
    if raw_data_dir.exists() and raw_data_dir.is_dir():
        logger.warning(f"检测到 VLM 数据缓存目录 '{raw_data_dir}'。正在删除以强制重新下载和加载。")
        try:
            shutil.rmtree(raw_data_dir)
            logger.info(f"VLM 数据缓存目录 '{raw_data_dir}' 已成功删除。")
        except Exception as e:
            logger.error(f"无法删除 VLM 数据缓存目录 '{raw_data_dir}'。请手动删除该目录或检查权限。错误详情: {e}")
            # 如果无法删除，尝试继续，但用户可能需要手动干预
    else:
        logger.info(f"VLM 数据缓存目录 '{raw_data_dir}' 不存在，无需清除。")

    # 使用 snapshot_download 下载数据集的原始文件
    # 对于 COCO，通常会下载 image 文件夹和 annotations 文件夹
    local_dataset_path = snapshot_download(
        repo_id=dataset_name,
        cache_dir=str(base_output_dir / 'raw_snapshots'),  # 将原始快照保存到另一个缓存目录
        local_dir=str(raw_data_dir),  # 指定复制到 raw_data_dir
        allow_patterns=["*train2014*", "*annotations*"]  # 限制只下载训练集和注解文件，减少下载量
    )
    logger.info(f"原始数据集文件已下载到: {local_dataset_path}")

    # 手动解析 COCO 注解文件并构建数据集
    # COCO 数据集通常包含 images/ 和 annotations/ 两个主要部分
    # 我们需要加载 captions_train2014.json 文件
    annotations_file = Path(raw_data_dir) / "annotations" / "captions_train2014.json"
    images_dir = Path(raw_data_dir) / "train2014"  # 图像文件所在的目录

    if not annotations_file.exists():
        logger.error(f"错误: COCO 注解文件未找到或路径不正确: {annotations_file}")
        logger.error("请确认 HuggingFaceM4/COCO 数据集结构或手动检查下载内容。")
        sys.exit(1)  # 无法找到注解文件，直接退出

    logger.info(f"正在解析 COCO 注解文件: {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images_metadata = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    # 构建一个简化的数据集列表
    dataset_list = []
    seen_image_ids = set()  # 用于限制处理的图像数量，对应 num_samples_to_process

    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in images_metadata:
            logger.warning(f"警告: 找到注解 (ID: {annotation['id']}) 对应图像 (ID: {image_id}) 的元数据缺失。跳过。")
            continue

        if num_samples_to_process > 0 and len(seen_image_ids) >= num_samples_to_process:
            break  # 达到样本限制

        image_info = images_metadata[image_id]
        image_filename = image_info['file_name']
        image_path = images_dir / image_filename

        if not image_path.exists():
            logger.warning(f"警告: 图像文件不存在: {image_path}。跳过。")
            continue

        # 避免重复的图像（因为一张图可能有多个字幕）
        if image_id not in seen_image_ids:
            try:
                # 尝试加载图像，如果图像有问题，会在这里报错
                with Image.open(image_path) as img:
                    img.load()  # 强制加载像素数据，检查有效性

                dataset_list.append({
                    "image": img,  # 直接存储 PIL Image 对象
                    "sentences_raw": [{"caption": annotation['caption']}],  # 保持原有的格式，方便 process_vlm_dataset
                    "image_id": image_id,
                    "original_image_path": str(image_path)  # 保留原始路径，方便调试
                })
                seen_image_ids.add(image_id)
            except Exception as e:
                logger.error(f"加载图像 {image_path} 失败: {e}。跳过该图像。")
                continue
        else:
            # 如果图像已经添加过，只更新它的字幕列表
            for item in dataset_list:
                if item["image_id"] == image_id:
                    item["sentences_raw"].append({"caption": annotation['caption']})
                    break

    # 从列表创建 datasets.Dataset
    dataset = Dataset.from_list(dataset_list)
    logger.info(f"已从本地文件加载并构建 {len(dataset)} 个 VLM 样本。")
    # --- VLM 数据下载和初始加载逻辑结束 ---

    logger.info("2. 正在处理 VLM 数据 (图像转换, 文本清洗等)...")
    processed_dataset = process_vlm_dataset(
        dataset,
        image_column='image',  # 现在直接从 'image' 列获取 PIL Image
        text_column=config['text_column'],  # 原始配置中的文本列名 'sentences_raw'
        distillation_config=config['distillation']
    )
    logger.info("VLM 数据处理完成。")

    logger.info("3. 正在进行 VLM 数据去重...")
    deduplicated_dataset = deduplicate_vlm_dataset(
        processed_dataset,
        image_column='image',  # 原始列名，仅为函数签名兼容性
        text_column='cleaned_captions'  # 确保使用清洗后的字幕进行去重
    )
    logger.info("VLM 数据去重完成。")

    logger.info("4. 正在进行 VLM 数据增强...")
    augmented_vlm_dataset = augment_vlm_dataset(
        deduplicated_dataset,
        image_column='image',
        text_column='cleaned_captions',
        augmentation_config=config.get('augmentation', {})
    )
    logger.info("VLM 数据增强步骤完成。")

    logger.info(f"5. 正在保存处理后的数据集到 '{processed_data_dir}'...")
    augmented_vlm_dataset.save_to_disk(str(processed_data_dir))
    logger.info(f"处理后的 VLM 数据集已保存到: {processed_data_dir}")

    logger.info("--- [Bedrock] VLM 数据处理流水线全部完成 ---")


if __name__ == "__main__":
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

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    text_config = project_root / "configs/data/text_pretrain.yaml"
    vlm_config = project_root / "configs/data/vlm_pretrain.yaml"

    if PIPELINE_TO_RUN in ["text", "all"]:
        run_text_pipeline(str(text_config))

    if PIPELINE_TO_RUN in ["vlm", "all"]:
        run_vlm_pipeline(str(vlm_config))

    logger.info("\n[Bedrock] 所有选定的流水线已成功执行。")