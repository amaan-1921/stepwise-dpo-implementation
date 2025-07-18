# train.py

from data.process_prm800k import load_raw_prm800k
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments
from reward_model import score_response_steps  # Your custom reward model
from stepwise_trainer import StepDPOTrainer
from torch.utils.data import random_split
import torch
import os

def main():
    # 1. Load and preprocess dataset
    dataset = load_raw_prm800k()
    dataset = dataset.rename_columns({"question": "prompt"})  # Required format

    # Optional: Split into train and eval (90-10)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # 2. Load tokenizer and reward model
    model_name = "mistralai/Mistral-7B-v0.1"  # or any compatible model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensure padding token is defined

    model = score_response_steps(model_name, tokenizer)  # Your custom model class

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints/step_dpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        # evaluation_strategy="epoch", 
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-6,
        warmup_steps=50,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=8,
        report_to="none",
    )

    # 4. Initialize custom StepDPOTrainer
    trainer = StepDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 5. Start training
    trainer.train()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
