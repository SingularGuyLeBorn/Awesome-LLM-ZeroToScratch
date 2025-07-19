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
    对单个文本字符串应用一系列确定性的清洗规则。
    此函数旨在标准化文本格式，移除无关内容，为后续处理步骤做准备。

    Args:
        text: 原始的文本字符串。

    Returns:
        清洗后的文本字符串。如果输入无效，则返回空字符串。
    """
    if not isinstance(text, str):
        logger.warning(f"检测到非字符串类型的数据，无法执行清洗，已跳过。数据类型: {type(text)}")
        return ""

    # 规则 1: 标准化空白字符。将多个空格、换行符、制表符替换为单个空格，并移除首尾空格。
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    # 规则 2: 移除维基百科文章中常见的元数据标题行 (例如, "= = = Section Title = = =")。
    # 这条规则是针对 Wikitext 数据集格式的特定优化。
    if cleaned_text.startswith('=') and cleaned_text.endswith('='):
        logger.debug(f"移除被识别为元数据行的文本: {cleaned_text[:80]}...")
        return ""  # 返回空字符串，该行将在后续的质量过滤中被移除

    return cleaned_text


def quality_filter(text: str) -> bool:
    """
    对文本字符串应用一套启发式规则，以评估其数据质量。
    此函数用于过滤掉低质量或无意义的文本。

    Args:
        text: 需要检查的文本字符串。

    Returns:
        如果文本通过所有质量检查，则返回 True，否则返回 False。
    """
    if not text:
        logger.debug("质量过滤：过滤掉空文本。")
        return False

    # 过滤规则 1: 最小长度（按词计数）。此规则可以有效过滤掉过短或无意义的行。
    # 这里的 "10" 是一个经验值，可以根据具体任务进行调整。
    if len(text.split()) < 10:
        logger.debug(f"质量过滤：因少于10个词而过滤掉的短文本: {text[:80]}...")
        return False

    # 过滤规则 2: 文本必须包含至少一个英文字母。此规则可以过滤掉纯数字或纯符号的行。
    if not re.search(r'[a-zA-Z]', text):
        logger.debug(f"质量过滤：因不含任何英文字母而被过滤的文本: {text[:80]}...")
        return False

    return True


def clean_text_dataset(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    将文本清洗和质量过滤函数应用于 Hugging Face DatasetDict 的所有分割。
    这是一个高级封装函数，它协调 `clean_text` 和 `quality_filter` 的执行。

    Args:
        dataset: 原始的 DatasetDict 对象。
        text_column: 包含待清洗文本的列名。

    Returns:
        一个新的 DatasetDict，其中包含了经过清洗和过滤的数据。
    """
    logger.info("开始对整个数据集进行文本清洗和质量过滤...")
    # 使用多进程以加速处理，进程数可以根据 CPU 核心数调整
    num_processes = os.cpu_count() if os.cpu_count() is not None else 1
    logger.info(f"使用 {num_processes} 个进程进行数据映射操作。")

    # 第一步: 应用 clean_text 函数进行初步清洗
    cleaned_dataset = dataset.map(
        lambda example: {text_column: clean_text(example[text_column])},
        num_proc=num_processes,
        desc="执行文本初步清洗"
    )
    logger.info("文本初步清洗完成。")

    # 第二步: 应用 quality_filter 函数进行质量过滤
    # 过滤掉在 clean_text 中可能返回空字符串的行，以及不符合质量标准的行
    filtered_dataset = cleaned_dataset.filter(
        lambda example: quality_filter(example[text_column]),
        num_proc=num_processes,
        desc="执行数据质量过滤"
    )
    
    # 记录每个分割的样本数量变化
    for split_name in dataset.keys():
        original_count = len(dataset[split_name])
        filtered_count = len(filtered_dataset[split_name])
        logger.info(f"'{split_name}' 分割：原始样本数 = {original_count}, 过滤后样本数 = {filtered_count} (移除了 {original_count - filtered_count} 个样本)")

    logger.info("数据集的文本清洗和质量过滤全部完成。")
    return filtered_dataset


def deduplicate_dataset(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    使用高效的、基于哈希的向量化方法，从 Hugging Face DatasetDict 的所有分割中删除精确的重复文本条目。

    Args:
        dataset: 要去重的 DatasetDict 对象。
        text_column: 包含要进行去重操作的文本的列名。

    Returns:
        一个新的 DatasetDict，其中重复的条目已被移除。
    """
    logger.info("开始对数据集进行高效的精确去重...")
    # 使用多进程以加速处理，进程数可以根据 CPU 核心数调整
    num_processes = os.cpu_count() if os.cpu_count() is not None else 1
    logger.info(f"使用 {num_processes} 个进程进行去重。")

    deduplicated_dataset = DatasetDict()
    for split_name, split_dataset in dataset.items():
        original_count = len(split_dataset)
        logger.info(f"正在处理 '{split_name}' 分割，共 {original_count} 个样本...")

        # 使用 map 方法高效地计算每个文本的哈希值
        # with_indices=True 会为每个样本提供一个唯一的索引，用于后续的筛选
        hashes = split_dataset.map(
            lambda example: {'hash': hash(example[text_column])},
            num_proc=num_processes,
            desc=f"为 '{split_name}' 分割计算哈希值"
        )

        seen_hashes = set()
        unique_indices = []

        # 迭代哈希值（这比迭代整个数据集要快得多）
        for i, h in enumerate(hashes['hash']):
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_indices.append(i)

        # 使用计算出的唯一索引来选择样本
        deduplicated_split = split_dataset.select(unique_indices)
        deduplicated_dataset[split_name] = deduplicated_split
        
        final_count = len(deduplicated_split)
        logger.info(
            f"'{split_name}' 分割去重完成。原始样本数: {original_count}, 去重后样本数: {final_count} (移除了 {original_count - final_count} 个重复项)"
        )

    logger.info("数据集精确去重完成。")
    return deduplicated_dataset


def augment_text_dataset(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    用于文本数据增强的概念性占位符函数。

    在真实的生产环境中，这里可以集成各种数据增强技术，例如：
    - **回译 (Back-translation)**: 将文本翻译到另一种语言再翻译回来，以产生新的表述。
    - **释义 (Paraphrasing)**: 使用语言模型生成与原文意思相同但措辞不同的句子。
    - **同义词替换**: 使用词库（如 WordNet）替换句子中的部分词语。
    - **生成式增强**: 使用大型语言模型根据原文生成新的、相关的文本样本。

    Args:
        dataset: 需要进行增强的 DatasetDict 对象。
        text_column: 包含待增强文本的列名。

    Returns:
        返回原始的 DatasetDict。这是一个占位符，需要实现具体逻辑才能实际增强数据。
    """
    logger.info("开始执行概念性文本数据增强 (当前为占位符逻辑)。")
    # -------------------------------------------------------------------
    # 警告: 下方是数据增强的示例代码，默认被注释掉。
    # 若要启用，请取消注释并安装所需库 (例如 `pip install nlpaug`)。
    # -------------------------------------------------------------------
    # from nlpaug.augmenter.word import SynonymAugmenter
    # aug = SynonymAugmenter(aug_src='wordnet')
    #
    # def _augment_example(example: Dict[str, Any]) -> Dict[str, Any]:
    #     original_text = example[text_column]
    #     # nlpaug 可能返回单个字符串或列表，确保处理逻辑的健壮性
    #     augmented_text_list = aug.augment(original_text)
    #     augmented_text = augmented_text_list[0] if isinstance(augmented_text_list, list) else augmented_text_list
    #     return {text_column: augmented_text if augmented_text else original_text}
    #
    # logger.info("正在应用同义词替换增强...")
    # num_processes = os.cpu_count() if os.cpu_count() is not None else 1
    # augmented_dataset = dataset.map(_augment_example, num_proc=num_processes, desc="应用数据增强")
    # logger.info("文本数据增强完成。")
    # return augmented_dataset
    # -------------------------------------------------------------------

    logger.warning("数据增强功能当前为占位符，未对数据进行实际修改。若需启用，请在 `augment_text_dataset` 函数中实现具体逻辑。")
    return dataset  # 按当前设计，返回未经修改的原始数据集

# END OF FILE: data_processing/process_text.py