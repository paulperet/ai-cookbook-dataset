# Guide: Preparing and Analyzing Data for Chat Model Fine-Tuning

This guide walks you through preparing and analyzing a chat dataset for fine-tuning a model like `gpt-3.5-turbo`. You'll learn how to validate the data format, compute key statistics, and estimate token counts for cost planning. The methods shown align with OpenAI's [current fine-tuning approach](https://platform.openai.com/docs/guides/fine-tuning).

> **Note:** For models like `babbage-002` or `davinci-002`, refer to the [legacy fine-tuning guide](https://platform.openai.com/docs/guides/legacy-fine-tuning).

## Prerequisites

Ensure you have the required Python libraries installed:

```bash
pip install tiktoken numpy
```

Then, import the necessary modules:

```python
import json
import tiktoken  # for token counting
import numpy as np
from collections import defaultdict
```

## Step 1: Load Your Dataset

First, load your chat dataset. This example uses a sample JSONL file, but you can replace the path with your own data.

```python
data_path = "data/toy_chat_fine_tuning.jsonl"

# Load the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)
```

**Expected Output:**
```
Num examples: 5
First example:
{'role': 'system', 'content': 'You are a happy assistant that puts a positive spin on everything.'}
{'role': 'user', 'content': 'I fell off my bike today.'}
{'role': 'assistant', 'content': "It's great that you're getting exercise outdoors!"}
```

## Step 2: Validate Data Format

To ensure your dataset meets the fine-tuning API requirements, run a series of format checks. The code below validates each conversation and categorizes any errors.

```python
# Format error checks
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue
        
    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue
        
    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1
        
        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1
        
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1
            
        content = message.get("content", None)
        function_call = message.get("function_call", None)
        
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1
    
    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")
```

If your dataset is correctly formatted, you should see:
```
No errors found
```

## Step 3: Define Token Counting Utilities

Token counting is essential for understanding dataset size and estimating costs. We'll use `tiktoken` with the `cl100k_base` encoding (used by `gpt-3.5-turbo` and `gpt-4`).

```python
encoding = tiktoken.get_encoding("cl100k_base")

# Simplified token counting (adapted from OpenAI Cookbook)
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
```

## Step 4: Analyze Dataset Warnings and Token Counts

Now, let's analyze the dataset for potential issues and compute token statistics.

```python
# Warnings and tokens counts
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
n_too_long = sum(l > 16385 for l in convo_lens)
print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")
```

**Sample Output:**
```
Num examples missing system message: 1
Num examples missing user message: 1

#### Distribution of num_messages_per_example:
min / max: 2, 9
mean / median: 3.8, 3.0
p5 / p95: 2.0, 6.6

#### Distribution of num_total_tokens_per_example:
min / max: 26, 8032
mean / median: 1648.4, 45.0
p5 / p95: 26.8, 4863.6

#### Distribution of num_assistant_tokens_per_example:
min / max: 4, 8000
mean / median: 1610.2, 10.0
p5 / p95: 6.0, 4811.2

0 examples may be over the 16,385 token limit, they will be truncated during fine-tuning
```

## Step 5: Estimate Fine-Tuning Cost

Finally, estimate the total tokens that will be billed during training. The fine-tuning process automatically adjusts the number of epochs based on your dataset size.

```python
# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 16385

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
```

**Sample Output:**
```
Dataset has ~4306 tokens that will be charged for during training
By default, you'll train for 20 epochs on this dataset
By default, you'll be charged for ~86120 tokens
```

Use the [OpenAI Pricing page](https://openai.com/pricing) to convert the token count into an estimated cost.

## Summary

You've successfully:
1. Loaded and inspected a chat dataset.
2. Validated its format against fine-tuning requirements.
3. Computed token counts and distributions.
4. Identified potential issues like missing messages or overly long conversations.
5. Estimated the tokens that will be billed during fine-tuning.

This analysis ensures your dataset is clean and helps you budget for the fine-tuning process.