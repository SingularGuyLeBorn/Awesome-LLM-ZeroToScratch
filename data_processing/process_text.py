# FILE: data_processing/process_text.py
"""
Bedrock Protocol: Module for text data cleaning and processing.

This module contains functions that perform deterministic, rule-based cleaning
on text datasets. The goal is to improve data quality, which is a cornerstone
of building robust models.
"""

import os
import logging
from typing import Dict, Any
from datasets import Dataset, DatasetDict
import re

# 配置日志
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Applies a series of cleaning rules to a single text string.

    Args:
        text: The raw text string.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        logger.warning(f"检测到非字符串类型数据进行清洗，已跳过：{type(text)}")
        return ""

    # Rule 1: Normalize whitespace. Replace multiple spaces, newlines, tabs with a single space.
    text = re.sub(r'\s+', ' ', text).strip()

    # Rule 2: Remove lines that are just navigation or metadata (common in wikitext).
    # This rule is specific to the Wikitext dataset structure.
    if text.startswith('=') and text.endswith('='):
        logger.debug(f"移除元数据行: {text[:50]}...")
        return ""

    return text


def quality_filter(text: str) -> bool:
    """
    Applies a set of quality heuristics to a text string.

    Args:
        text: The text string to check.

    Returns:
        True if the text passes quality checks, False otherwise.
    """
    if not text:
        logger.debug("过滤空文本。")
        return False

    # Filter 1: Minimum length (in words). Prevents very short or empty lines.
    if len(text.split()) < 10:
        logger.debug(f"过滤短文本 (少于10个词): {text[:50]}...")
        return False

    # Filter 2: Must contain at least one letter. Filters out purely numerical or symbol-only lines.
    if not re.search(r'[a-zA-Z]', text):
        logger.debug(f"过滤不含字母的文本: {text[:50]}...")
        return False

    return True


def clean_text_dataset(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    Applies cleaning and filtering to all splits of a Hugging Face DatasetDict.

    Args:
        dataset: The raw DatasetDict object.
        text_column: The name of the column containing the text to clean.

    Returns:
        A new DatasetDict with the cleaned and filtered data.
    """
    logger.info("开始对数据集进行文本清洗和质量过滤...")
    num_processes = os.cpu_count() if os.cpu_count() is not None else 1
    logger.info(f"使用 {num_processes} 个进程进行数据映射。")

    cleaned_dataset = dataset.map(
        lambda example: {text_column: clean_text(example[text_column])},
        num_proc=num_processes,
        desc="执行文本清洗"
    )
    logger.info("文本清洗完成。")

    # Apply quality filter after initial cleaning
    filtered_dataset = cleaned_dataset.filter(
        lambda example: quality_filter(example[text_column]),
        num_proc=num_processes,
        desc="执行质量过滤"
    )
    logger.info(f"质量过滤完成。原始样本数: {len(dataset['train'])}, 过滤后样本数: {len(filtered_dataset['train'])}")

    return filtered_dataset


def deduplicate_dataset(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    Removes exact duplicate text entries from all splits of a Hugging Face DatasetDict.

    Args:
        dataset: The DatasetDict object to deduplicate.
        text_column: The name of the column containing the text to deduplicate.

    Returns:
        A new DatasetDict with duplicate entries removed.
    """
    logger.info("开始对数据集进行精确去重...")
    deduplicated_dataset = DatasetDict()

    for split_name, split_dataset in dataset.items():
        original_count = len(split_dataset)
        seen_texts = set()
        unique_indices = []
        for i, example in enumerate(split_dataset):
            text = example[text_column]
            if text not in seen_texts:
                seen_texts.add(text)
                unique_indices.append(i)

        deduplicated_split = split_dataset.select(unique_indices)
        deduplicated_dataset[split_name] = deduplicated_split
        logger.info(
            f"拆分 '{split_name}' 去重完成。原始样本数: {original_count}, 去重后样本数: {len(deduplicated_split)} (移除了 {original_count - len(deduplicated_split)} 个重复项)")

    logger.info("数据集精确去重完成。")
    return deduplicated_dataset


def augment_text_dataset(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    Conceptual function for augmenting text data.
    In a real scenario, this would involve techniques like back-translation,
    paraphrasing, or generative augmentation.

    Args:
        dataset: The DatasetDict object to augment.
        text_column: The name of the column containing the text to augment.

    Returns:
        The DatasetDict, potentially with augmented examples.
    """
    logger.info("开始执行概念性文本数据增强 (目前为占位符逻辑)。")
    # 实际的数据增强逻辑将在这里实现。
    # 例如，你可以添加如下代码：
    # from nlpaug.augmenter.word import SynonymAugmenter
    # aug = SynonymAugmenter(aug_src='wordnet')
    #
    # def _augment_example(example: Dict[str, Any]) -> Dict[str, Any]:
    #     original_text = example[text_column]
    #     augmented_text = aug.augment(original_text) # 假设返回一个字符串
    #     # 你可以选择将增强后的文本添加到新列，或替换现有列，或增加新样本
    #     return {text_column: augmented_text if augmented_text else original_text}
    #
    # augmented_dataset = dataset.map(_augment_example, num_proc=os.cpu_count())
    # return augmented_dataset

    logger.warning("数据增强功能目前仅为占位符。若需实际功能，请在此处添加具体增强逻辑。")
    logger.info("概念性文本数据增强完成。数据集未实际修改 (除非你添加了逻辑)。")
    return dataset  # 目前返回原始数据集

# END OF FILE: data_processing/process_text.py