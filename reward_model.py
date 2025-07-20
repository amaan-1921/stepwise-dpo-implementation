#reward_model.py
from openai import OpenAI
import os
import time
from dotenv import load_dotenv
import torch
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
from transformers import GPT2LMHeadModel
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn


load_dotenv()

class RewardModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2Model.from_pretrained("gpt2")
        self.score = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0][:, -1, :]  # Use last token embedding for GPT2
        score = self.score(hidden_states)
        return score

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)


# Create client with API key (either from environment or passed explicitly)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Call GPT-4 to score a single reasoning step
def score_reasoning_step(prompt: str, step: str) -> int:
    system_message = "You are a helpful assistant that evaluates reasoning steps for their accuracy and relevance."

    user_message = f"""\
Task: Rate the helpfulness of the following reasoning step in answering the question.

Question:
{prompt}

Step:
"{step}"

Is this reasoning step helpful and accurate in answering the question? Reply only with 'Yes' or 'No'."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
        )

        answer = response.choices[0].message.content.strip() # type: ignore
        if "Yes" in answer:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error scoring step: {e}")
        return 0  # Return 0 if API fails

# Score all steps in a multi-step response
def score_response_steps(prompt: str, steps: list[str]) -> list[int]:
    scores = []
    for step in steps:
        score = score_reasoning_step(prompt, step)
        scores.append(score)
        time.sleep(1.5)  # To stay under rate limits
    return scores
