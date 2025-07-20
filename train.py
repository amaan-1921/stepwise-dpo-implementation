# train.py

# --- Imports ---
import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from stepwise_trainer import StepDPOTrainer
from data.process_prm800k import load_raw_prm800k


# --- Custom Data Collator ---
@dataclass
class CustomDPODataCollator:
    """A custom data collator for DPO that pads all fields correctly."""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Create lists of the unpadded input_ids and attention_masks
        prompt_ids = [f["prompt_input_ids"] for f in features]
        prompt_masks = [f["prompt_attention_mask"] for f in features]
        chosen_ids = [f["chosen_input_ids"] for f in features]
        chosen_masks = [f["chosen_attention_mask"] for f in features]
        rejected_ids = [f["rejected_input_ids"] for f in features]
        rejected_masks = [f["rejected_attention_mask"] for f in features]

        # Pad each of the sections separately
        padded_prompts = self.tokenizer.pad(
            {"input_ids": prompt_ids, "attention_mask": prompt_masks}, return_tensors="pt", padding=True
        )
        padded_chosen = self.tokenizer.pad(
            {"input_ids": chosen_ids, "attention_mask": chosen_masks}, return_tensors="pt", padding=True
        )
        padded_rejected = self.tokenizer.pad(
            {"input_ids": rejected_ids, "attention_mask": rejected_masks}, return_tensors="pt", padding=True
        )

        # Combine them into a single dictionary for the batch
        batch = {
            "prompt_input_ids": padded_prompts["input_ids"],
            "prompt_attention_mask": padded_prompts["attention_mask"],
            "chosen_input_ids": padded_chosen["input_ids"],
            "chosen_attention_mask": padded_chosen["attention_mask"],
            "rejected_input_ids": padded_rejected["input_ids"],
            "rejected_attention_mask": padded_rejected["attention_mask"],
        }
        return batch


# --- Main Training Function ---
def main():
    # 1. Load Dataset
    dataset = load_raw_prm800k()
    dataset = dataset.rename_columns({"question": "prompt"})

    # 2. Load Tokenizer and Model
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3. CRITICAL: Verify and fix vocabulary size mismatch
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"Model embedding size: {model_vocab_size}")
    if tokenizer_vocab_size != model_vocab_size:
        print("!!! MISMATCH DETECTED !!! Resizing model embeddings to match tokenizer.")
        model.resize_token_embeddings(tokenizer_vocab_size)

    # 4. Define Data Processing Functions
    def tokenize_dpo_dataset(examples):
        """Tokenizes the prompt, chosen, and rejected fields."""
        prompt_tokens = tokenizer(examples["prompt"], truncation=True)
        chosen_tokens = tokenizer(examples["chosen"], truncation=True)
        rejected_tokens = tokenizer(examples["rejected"], truncation=True)
        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }

    def has_invalid_ids(example, vocab_size):
        """Check if any token ID in the example is out of the vocab range."""
        all_ids = (
            example["prompt_input_ids"] + 
            example["chosen_input_ids"] + 
            example["rejected_input_ids"]
        )
        for token_id in all_ids:
            if not (0 <= token_id < vocab_size):
                return True
        return False

    # 5. Process and Filter Dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_dpo_dataset, batched=True, remove_columns=list(dataset.features))

    print("Filtering out empty responses...")
    filtered_dataset = tokenized_dataset.filter(
        lambda example: len(example["chosen_input_ids"]) > 1 and len(example["rejected_input_ids"]) > 1
    )

    print("Validating token IDs...")
    final_dataset = filtered_dataset.filter(
        lambda example: not has_invalid_ids(example, len(tokenizer))
    )

    print("Splitting dataset...")
    split_dataset = final_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints/step_dpo_final",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
        report_to="none",
        remove_unused_columns=False, # Crucial for custom loss function
    )

    # 7. Initialize Trainer
    data_collator = CustomDPODataCollator(tokenizer=tokenizer)
    
    trainer = StepDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 8. Train
    print("Starting training...")
    trainer.train()

# --- Script Entrypoint ---
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()