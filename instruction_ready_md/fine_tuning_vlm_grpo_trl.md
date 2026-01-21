# Post-Training a Vision Language Model for Reasoning with GRPO using TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

🚨 **WARNING**: This guide is resource-intensive and requires substantial computational power. If you're running this in a Colab environment, it will utilize an A100 GPU.

In this tutorial, you will learn how to post-train a [Vision Language Model (VLM)](https://huggingface.co/blog/vlms-2025) using [Group Relative Policy Optimization (GRPO)](https://huggingface.co/docs/trl/grpo_trainer) to add reasoning capabilities. We'll use the Hugging Face ecosystem, specifically the [Transformer Reinforcement Learning library (trl)](https://huggingface.co/docs/trl/index).

We'll fine-tune [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on a subset of the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. This dataset contains images with problem descriptions, solutions, and the reasoning traces used to reach those solutions. By leveraging this data format and custom GRPO reward functions, we will teach the model how to reason step-by-step to find a solution.

## 1. Setup and Installation

First, install the required libraries. We'll install `trl` from source, as the VLM GRPO trainer is not yet included in an official release.

```bash
pip install -U -q git+https://github.com/huggingface/trl.git peft math_verify qwen-vl-utils[decord]
```

Authenticate with your Hugging Face account to save and share the trained model.

```python
from huggingface_hub import login

login()
```

## 2. Load and Prepare the Dataset

We'll use the `lmms-lab/multimodal-open-r1-8k-verified` dataset, which contains 8k multimodal examples focused on math reasoning. For this tutorial, we'll use only 5% of the data and split it into training and test sets to speed up the process.

```python
from datasets import load_dataset

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
dataset = load_dataset(dataset_id, split='train[:5%]')

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']
```

Let's inspect the dataset structure.

```python
print(train_dataset)
```

The dataset has the following columns: `image`, `problem`, `solution`, `original_question`, and `original_answer`.

### 2.1 Format Data for the GRPO Trainer

The GRPO trainer expects data in a specific conversational format. We'll create a system prompt (inspired by DeepSeek R1) and convert each sample into a conversation that includes the system prompt, the image, and the problem description.

We also set the tokenizer's `padding_side` to `"left"` to ensure generated completions are concatenated directly after the prompt, which is essential for GRPO's token-level probability comparisons.

```python
from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["image"],
    }

train_dataset = train_dataset.map(make_conversation)
```

Now, let's examine a formatted example.

```python
print(train_dataset[0]['prompt'])
```

Finally, remove the columns we no longer need for training.

```python
train_dataset = train_dataset.remove_columns(['problem', 'original_question', 'original_answer'])
print(train_dataset)
```

The dataset now contains only `image`, `solution`, and `prompt`.

## 3. Post-Training the VLM with GRPO

GRPO (Group Relative Policy Optimization) is a reinforcement learning method that removes the need for a separate value model, simplifying the training pipeline compared to PPO. We'll implement the training using `trl`'s `GRPOConfig` and `GRPOTrainer`.

### 3.1 Load the Baseline Model

Load the `Qwen2.5-VL-3B-Instruct` model.

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### 3.2 Configure LoRA for Efficient Fine-Tuning

We'll use Low-Rank Adaptation (LoRA) to fine-tune the model efficiently, targeting the `q_proj` and `v_proj` modules.

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

### 3.3 Define Reward Functions

GRPO uses reward functions to guide the model's behavior. We'll implement two reward functions, adapted from the Open R1 project:

1.  **Format Reward:** Ensures the model's output follows the required `<think>...</think><answer>...</answer>` structure.
2.  **Accuracy Reward:** Checks if the model's answer matches the ground truth solution, using mathematical verification when possible.

```python
import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards
```

### 3.4 Configure GRPO Training Parameters

Set up the training configuration. The parameters below are adjusted for a Colab environment. For a production run, you would increase `num_generations`, use a larger model, and train on the full dataset.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="Qwen2.5-VL-3B-Instruct-Thinking",
    learning_rate=1e-5,
    remove_unused_columns=False, # Required to access the `solution` column in accuracy_reward
    num_train_epochs=1,
    bf16=True,

    # Data preprocessing parameters
    per_device_train_batch_size=2,
    max_completion_length=1024,
    num_generations=2,
    max_prompt_length=2048,

    # Reporting and saving parameters
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)
```

### 3.5 Initialize the GRPO Trainer and Start Training

Create the `GRPOTrainer` with the model, processor, reward functions, training arguments, and dataset.

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)
```

Finally, start the training process.

```python
trainer.train()
```

During training, you will see logs showing the training loss. After training completes, your model will be saved to the specified output directory and, if configured, pushed to the Hugging Face Hub.

## Summary

In this tutorial, you learned how to post-train a Vision Language Model to improve its reasoning capabilities using GRPO. The key steps were:
1.  Installing dependencies and authenticating.
2.  Loading and formatting a multimodal dataset for instruction following.
3.  Loading a base VLM and configuring LoRA for efficient fine-tuning.
4.  Defining custom reward functions to enforce output format and solution accuracy.
5.  Configuring GRPO training parameters and running the training loop.

This approach provides a foundation for teaching VLMs to produce structured, step-by-step reasoning, which is crucial for complex problem-solving tasks.