import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trl import DPOTrainer
import torch

# ✅ Import your own reward model
from reward_model import score_response_steps

class StepDPOTrainer(DPOTrainer):
    def __init__(self, *args, reward_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use passed reward model or fallback to default score_response_steps
        self.reward_model = reward_model or score_response_steps

    def compute_step_rewards(self, prompts, completions):
        rewards = []
        for prompt, steps in zip(prompts, completions):
            step_rewards = self.reward_model(prompt, steps)
            rewards.append(step_rewards)
        return rewards

    def compute_loss(self, model, inputs, return_outputs=False):
        prompts = inputs["prompt"]
        chosen = inputs["chosen"]
        rejected = inputs["rejected"]

        chosen_rewards = self.compute_step_rewards(prompts, chosen)
        rejected_rewards = self.compute_step_rewards(prompts, rejected)

        # Aggregate rewards (sum over steps)
        chosen_scores = torch.tensor([sum(r) for r in chosen_rewards])
        rejected_scores = torch.tensor([sum(r) for r in rejected_rewards])

        loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()

        return (loss, None) if return_outputs else loss
