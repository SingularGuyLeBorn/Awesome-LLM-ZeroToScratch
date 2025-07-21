# FILE: diagnose_dataset.py
"""
Bedrock Protocol: Dataset Structure Diagnostic Tool.

This script loads a specified dataset from Hugging Face and prints the
structure of the first example. It's a simple but powerful tool to
verify data formats before writing complex processing logic.
"""

from datasets import load_dataset
import sys


def diagnose(dataset_name: str, split: str = "train", num_samples_to_show: int = 1):
    """
    Loads a dataset and prints detailed information about the first few samples.

    Args:
        dataset_name: The name of the dataset on the Hugging Face Hub.
        split: The dataset split to inspect (e.g., "train", "test").
        num_samples_to_show: The number of samples to print.
    """
    print(f"--- [Bedrock] Starting Dataset Diagnosis for: {dataset_name} ---")

    try:
        # Load the dataset
        print(f"\nAttempting to load dataset '{dataset_name}', split '{split}'...")
        dataset = load_dataset(dataset_name, split=split)
        print("Dataset loaded successfully.")

        if len(dataset) < num_samples_to_show:
            print(
                f"Warning: Dataset has only {len(dataset)} samples, which is less than requested {num_samples_to_show}.")
            num_samples_to_show = len(dataset)

        print(f"\nInspecting the first {num_samples_to_show} sample(s)...")
        print("=" * 50)

        for i in range(num_samples_to_show):
            # Get the first sample
            sample = dataset[i]

            print(f"\n--- SAMPLE #{i + 1} ---")

            # List all keys in the sample
            print(f"Available keys: {list(sample.keys())}")

            # Inspect the fields relevant to DPO
            dpo_keys = ["prompt", "chosen", "rejected"]

            for key in dpo_keys:
                if key in sample:
                    value = sample[key]
                    value_type = type(value)

                    print(f"\nField: '{key}'")
                    print(f"  - Type: {value_type}")
                    print(f"  - Content:\n---\n{value}\n---")
                else:
                    print(f"\nField: '{key}' - NOT FOUND IN SAMPLE.")

            print("=" * 50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check if the dataset name and split are correct.")


if __name__ == "__main__":
    # You can change the dataset name here for other diagnostics
    # Defaulting to the one causing issues in DPO.
    target_dataset = "trl-internal-testing/hh-rlhf-trl-style"

    if len(sys.argv) > 1:
        target_dataset = sys.argv[1]

    diagnose(target_dataset)

# END OF FILE: diagnose_dataset.py