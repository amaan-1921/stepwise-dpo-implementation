from openai import OpenAI
import os
import time
from dotenv import load_dotenv
import torch
from transformers import AutoModelForSequenceClassification


load_dotenv()

class RewardModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # for scalar score
        )

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits 

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
