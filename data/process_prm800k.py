# data/process_prm800k.py

from datasets import load_dataset, Dataset
from typing import Dict, List, cast
import pprint


def load_raw_prm800k() -> Dataset:
    """
    Loads the Intel/orca_dpo_pairs (prm800k) dataset from Hugging Face.
    Returns the raw Hugging Face Dataset object.
    """
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    return cast(Dataset, dataset)


def preview_samples(dataset: Dataset, num_samples: int = 3):
    """
    Prints out a few prompt-chosen-rejected examples.
    """
    print(f"\nShowing {num_samples} samples from prm800k:\n")
    for i in range(num_samples):
        sample = dataset[i]
        print(f"[Prompt {i + 1}]:\n{sample['question']}\n")
        print(f"[Chosen]:\n{sample['chosen'][:300]}...\n")
        print(f"[Rejected]:\n{sample['rejected'][:300]}...\n")
        print("=" * 80)


if __name__ == "__main__":
    dataset = load_raw_prm800k()
    preview_samples(dataset)
