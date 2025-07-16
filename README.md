# stepwise-dpo-implementation
Implementation of Stepwise DPO with LLM-based rewards for Future AGI assignment.

## Objectives
- Build an LLM-based step-wise reward model
- Subclass Hugging Face's DPOTrainer with step-level aggregation
- (Bonus) Train and evaluate a small model

## Dataset
Preferred: `prm800k`

## Directory Structure
- `src/`: core code (trainer, reward model, training script)
- `data/`: dataset processing
- `results/`: logs, metrics, output models
- `notebooks/`: for experiment tracking and EDA'

## End-to-End Flow

1. [Prompt]
        ↓
2. [Base Model] → generates → [Response A] and [Response B]
        ↓
3. [LLM Reward Model] → scores each step in A and B individually
        ↓
4. [StepwiseDPOTrainer] → compares scores → applies preference-based loss
        ↓
5. [Fine-tuned Model] learns to prefer better stepwise answers

## Setup
```bash
pip install -r requirements.txt
```


