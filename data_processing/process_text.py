# FILE: data_processing/process_text.py
"""
Bedrock Protocol: Module for text data cleaning and processing.

This module contains functions that perform deterministic, rule-based cleaning
on text datasets. The goal is to improve data quality, which is a cornerstone
of building robust models.
"""

import os
from typing import Dict, Any
from datasets import Dataset, DatasetDict
import re


def clean_text(text: str) -> str:
    """
    Applies a series of cleaning rules to a single text string.

    Args:
        text: The raw text string.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Rule 1: Normalize whitespace. Replace multiple spaces, newlines, tabs with a single space.
    text = re.sub(r'\s+', ' ', text).strip()

    # Rule 2: Remove lines that are just navigation or metadata (common in wikitext).
    # This rule is specific to the Wikitext dataset structure.
    if text.startswith('=') and text.endswith('='):
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
        return False

    # Filter 1: Minimum length (in words). Prevents very short or empty lines.
    if len(text.split()) < 10:
        return False

    # Filter 2: Must contain at least one letter. Filters out purely numerical or symbol-only lines.
    if not re.search(r'[a-zA-Z]', text):
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

    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single example from the dataset."""
        cleaned = clean_text(example[text_column])
        return {
            text_column: cleaned
        }

    # Mandate of Empirical Proof: Processing steps are applied deterministically.
    # Use .map() for efficient, parallelizable processing.
    # num_proc is set to os.cpu_count() for better resource utilization.
    num_processes = os.cpu_count() if os.cpu_count() is not None else 1
    print(f"Using {num_processes} processes for data mapping.")

    cleaned_dataset = dataset.map(
        process_example,
        num_proc=num_processes
    )

    # Apply quality filter after initial cleaning
    filtered_dataset = cleaned_dataset.filter(
        lambda example: quality_filter(example[text_column]),
        num_proc=num_processes
    )

    return filtered_dataset

# END OF FILE: data_processing/process_text.py