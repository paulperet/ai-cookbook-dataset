# Post-Training an LLM for Reasoning with GRPO in TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

This guide walks you through post-training a Large Language Model (LLM) using **Group Relative Policy Optimization (GRPO)**, a reinforcement learning technique introduced in the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300). GRPO is designed to scale test-time compute for extended reasoning, making it highly effective for complex tasks like mathematical problem-solving.

The technique is available via the [TRL library](https://huggingface.co/docs/trl/main/en/grpo_trainer#quick-start). For a deeper dive into the full training pipeline used for models like **DeepSeek-R1**, check out the [Open-R1 project](https://github.com/huggingface/open-r1).

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install -U -q trl peft math_verify
# Tested with transformers==4.47.1, trl==0.14.0, datasets==3.2.0, peft==0.14.0, accelerate==1.2.1, math_verify==0.3.3
```

Authenticate with Hugging Face to save and share your model.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Load and Prepare the Dataset

GRPO excels at tasks requiring multi-step reasoning. We'll use the [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) dataset, which contains mathematical problems paired with detailed solution steps.

```python
from datasets import load_dataset

dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:5%]', 'test[:5%]'])
```

Let's inspect the dataset structure and a sample.

```python
print(train_dataset)
```

```python
print(train_dataset[0])
```

### Format the Data for Conversational Training

The DeepSeek-R1 training uses a specific system prompt to structure reasoning. We'll adapt our dataset to follow this format, prompting the model to first think through the problem and then provide an answer within designated tags.

Define the system prompt.

```python
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
```

Create a function to format each example into a conversational prompt.

```python
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)
```

Verify the formatted prompt.

```python
print(train_dataset[0]['prompt'])
```

Now, remove unnecessary columns, keeping only the custom `prompt` and the `solution` for verification.

```python
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print(train_dataset)
```

## Step 2: Load and Configure the Base Model

We'll use [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as our base model. It's lightweight for demonstration, but for production, consider a larger model.

```python
import torch
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

### Apply LoRA for Efficient Fine-Tuning

We'll use Low-Rank Adaptation (LoRA) to fine-tune the model efficiently, updating only a small subset of parameters.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## Step 3: Define Reward Functions

GRPO uses reward functions to guide training. We'll implement two: one for format compliance and one for answer accuracy.

### 1. Format Reward

This function checks if the model's output correctly uses the `<think>` and `<answer>` tags.

```python
import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list
```

### 2. Accuracy Reward

This function uses the `math_verify` library to compare the model's extracted answer against the ground truth solution.

```python
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion matches the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards
```

## Step 4: Configure GRPO Training

Now, set up the GRPO training configuration. We'll use a short training run with reduced generation lengths for demonstration.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False, # Required to access the 'solution' column in the reward function
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,

    # Data preprocessing parameters
    max_completion_length=64,  # Reduced from default 256 for speed
    num_generations=2,         # Number of completions to sample per prompt
    max_prompt_length=512,     # Maximum length of the input prompt

    # Generation parameters
    temperature=1.0,
    top_p=0.95,

    # Reward weighting
    reward_funcs=[format_reward, accuracy_reward],
    reward_funcs_weights=[0.5, 0.5], # Equal weight for format and accuracy
)
```

**Key Parameters Explained:**
*   `max_completion_length`: Maximum tokens the model can generate per response.
*   `num_generations`: How many candidate responses are sampled per training step for reward comparison.
*   `max_prompt_length`: Truncates prompts longer than this value.
*   `reward_funcs`: List of your defined reward functions.
*   `reward_funcs_weights`: Relative importance of each reward function.

## Step 5: Initialize the Trainer and Start Training

Create the GRPO Trainer with your model, tokenizer, datasets, and configuration.

```python
from transformers import AutoTokenizer
from trl import GRPOTrainer

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set padding token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Begin training
trainer.train()
```

## Step 6: Save the Trained Model

Once training is complete, save your fine-tuned model and its tokenizer.

```python
model.save_pretrained("qwen2-0.5b-grpo-math")
tokenizer.save_pretrained("qwen2-0.5b-grpo-math")
```

You can now push the model to the Hugging Face Hub for sharing.

```python
model.push_to_hub("your-username/qwen2-0.5b-grpo-math")
tokenizer.push_to_hub("your-username/qwen2-0.5b-grpo-math")
```

## Summary

You have successfully post-trained a language model using the GRPO algorithm via TRL. This guide covered:
1.  Preparing a reasoning-focused dataset in a conversational format.
2.  Applying LoRA for parameter-efficient fine-tuning.
3.  Defining custom reward functions for format and accuracy.
4.  Configuring and executing GRPO training.

To improve results, consider training for more epochs, using a larger base model, increasing `max_completion_length`, and fine-tuning the reward function weights. For a production-grade pipeline, refer to the [Open-R1 project](https://github.com/huggingface/open-r1).