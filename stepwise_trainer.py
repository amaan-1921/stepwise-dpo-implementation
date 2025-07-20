# stepwise_trainer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers.trainer import Trainer
import torch
from trl import DPOTrainer
from typing import List
from torch import device

# ✅ Import your custom stepwise reward function
from reward_model import score_response_steps


class StepDPOTrainer(DPOTrainer):
    def __init__(self, *args, reward_model=None, **kwargs):
        """
        Extends DPOTrainer to use stepwise reward computation.
        """
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model

    def compute_step_rewards(
    self,
    model,
    prompt_input_ids,
    prompt_attention_mask,
    response_input_ids,
    response_attention_mask
) -> List[List[float]]:
        """
        Computes stepwise logprobs (rewards) for each response token given the prompt.
        """
        # Concatenate prompt + response
        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)

        # Get logits from model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print("Model output type:", type(outputs))
        vocab_size = self.model.config.vocab_size
        invalid_ids = input_ids[(input_ids < 0) | (input_ids >= vocab_size)]
        if invalid_ids.numel() > 0:
            print("!!! FATAL: Invalid token ID detected right before model call !!!")
            print(f"Invalid IDs: {invalid_ids.tolist()}")
            print(f"Batch shape: {input_ids.shape}")
            raise ValueError("Execution stopped due to data corruption.")
        logits = outputs.logits  # shape: [batch_size, total_seq_len, vocab_size]

        total_seq_len = input_ids.size(1)
        response_len = response_input_ids.size(1)

        # Extract logits only for the response tokens
        response_logits = logits[:, -response_len-1:-1, :]  # shift by 1

        # Apply log_softmax to get log-probabilities over vocab
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)  # shape: [batch, response_len, vocab]

        # Now get logprobs of the actual generated tokens
        token_logprobs = log_probs.gather(2, response_input_ids.unsqueeze(-1)).squeeze(-1)
        # shape: [batch, response_len]

        # Return a list of lists
        return token_logprobs # shape: [batch, response_len]


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes DPO loss using sum of stepwise rewards.
        """
        print("KEYS IN BATCH:", list(inputs.keys()))
        prompt_input_ids = inputs["prompt_input_ids"]
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]

        prompt_attention_mask = inputs["prompt_attention_mask"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_attention_mask = inputs["rejected_attention_mask"]

        # Step 2: Compute rewards using tokenized inputs
        
        chosen_rewards = self.compute_step_rewards(model, prompt_input_ids, prompt_attention_mask, chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.compute_step_rewards(model, prompt_input_ids, prompt_attention_mask, rejected_input_ids, rejected_attention_mask)
        device = next(model.parameters()).device
        # Step 3: Convert stepwise reward lists to tensors
        chosen_scores = torch.sum(chosen_rewards, dim=1)
        rejected_scores = torch.sum(rejected_rewards, dim=1)

        # Step 4: DPO loss calculation
        loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()

        return (loss, None) if return_outputs else loss
