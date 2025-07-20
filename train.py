# train.py

from data.process_prm800k import load_raw_prm800k
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from torch.utils.data import random_split
from reward_model import RewardModel  # Your custom reward model wrapper
from stepwise_trainer import StepDPOTrainer  # Your custom trainer
import torch
import os

def main():
    # 1. Load and preprocess dataset
    dataset = load_raw_prm800k()
    dataset = dataset.rename_columns({"question": "prompt"})  # Ensure proper format

    # Optional: Split into train and eval sets
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # 2. Load tokenizer and base model (distilgpt2)
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Add padding if missing

    # 3. Initialize custom reward model (wraps distilgpt2)
    model = RewardModel.from_pretrained(model_name)

    # 4. Define training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints/step_dpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-6,
        warmup_steps=50,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        gradient_accumulation_steps=8,
        report_to="none",  # Disable reporting to wandb/huggingface
    )

    # 5. Initialize trainer
    trainer = StepDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Include tokenizer if your trainer accepts it
    )

    # 6. Train the model
    trainer.train()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()