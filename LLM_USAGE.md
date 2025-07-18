---

### `LLM_USAGE.md` (Initial Content)
# LLM_USAGE.md

> This file tracks all usage of LLMs (e.g., ChatGPT, Claude, Gemini 2.5 Pro) for assistance during this project.

---

### 2025-07-15
- Used ChatGPT to break down the assignment and interpret requirements.
- Also utlized ChatGPT to understand what is Direct Preference Optimization (DPO) and its advantages over RLHF.
- Suggested how to organize and track AI usage through this file.

---

### 2025-07-16
- Designed folder structure and initial README with assistance.
- Generated initial content for README.md.
- Used ChatPDF to summarize the paper "Generative Verifiers: Reward Modeling as Next-Token Prediction", which was provided for reference in the assignment.
- Used ChatGPT to suggest required packages for project setup.
- Added core libraries to `requirements.txt` for dataset loading, OpenAI usage, and trainer utilities.
- Was assisted to update .gitignore to exclude cache, envs, results, and reference code.
- Used ChatGPT to describe and visualize full end-to-end Stepwise DPO workflow.
- Used ChatGPT to generate `process_prm800k.py` for loading and previewing prm800k dataset.
- Helped me with an exploration notebook `explore_prm800k.ipynb` to inspect structure and test step-splitting.

---


### 2025-07-17

- Used ChatGPT to revisit the problem statement and deeply understand the LLM based reward system.
- Took help with reward_model.py to understand how to implement an LLM-based scoring system for reasoning steps.
- Helped me create a script for testing the openAI API.
- Used ChatGPT to help me analyze the baseline code that was given in the document. Currently focusing on `train.py` and `stepdpo_trainer.py`

---

### 2025-07-18
- Used ChatGPT to understand how to `compute_loss` function works.
- Helped me understand how StepDPOTrainer works with Trainer.
- Helped me code `stepwise_trainer.py`.
- Used ChatGPT to debug issues in `stepwise_trainer.py`.