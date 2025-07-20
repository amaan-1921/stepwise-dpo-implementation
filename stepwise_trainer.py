# stepwise_trainer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from trl import DPOTrainer

# ✅ Import your custom stepwise reward function
from reward_model import score_response_steps


class StepDPOTrainer(DPOTrainer):
    def __init__(self, *args, reward_model=None, **kwargs):
        """
        Extends DPOTrainer to use stepwise reward computation.
        """
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model or score_response_steps

    def compute_step_rewards(self, prompts, completions):
        """
        Computes stepwise rewards for each (prompt, response) pair.
        """
        rewards = []
        for prompt, steps in zip(prompts, completions):
            step_rewards = self.reward_model(prompt, steps)
            if isinstance(step_rewards, torch.Tensor):
                step_rewards = step_rewards.tolist()
            rewards.append(step_rewards)
        return rewards

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes DPO loss using sum of stepwise rewards.
        """
        prompts = inputs["prompt"]
        chosen = inputs["chosen"]
        rejected = inputs["rejected"]

        chosen_rewards = self.compute_step_rewards(prompts, chosen)
        rejected_rewards = self.compute_step_rewards(prompts, rejected)

        # Convert stepwise reward lists to tensors
        device = model.device
        chosen_scores = torch.tensor([sum(r) for r in chosen_rewards], device=device, dtype=torch.float32)
        rejected_scores = torch.tensor([sum(r) for r in rejected_rewards], device=device, dtype=torch.float32)

        # DPO loss: encourage higher reward for chosen over rejected
        loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()

        return (loss, None) if return_outputs else loss
