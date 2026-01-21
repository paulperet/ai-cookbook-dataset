# Efficient Online Training with GRPO and vLLM in TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

## Introduction

Online training methods like **Group Relative Policy Optimization (GRPO)** require the model to generate outputs in real time during training. This "online" generation is often a critical bottleneck, as it is both compute- and memory-intensive, especially for large language models (LLMs).

This guide demonstrates how to overcome this bottleneck by combining **vLLM**, a high-throughput, low-latency inference engine, with **TRL**. On a single GPU, TRL and vLLM can share resources efficiently for faster training. In larger multi-GPU setups, vLLM can run as a separate process on dedicated GPUs while TRL handles training on others, enabling seamless scaling.

Although we focus on GRPO, this setup is compatible with any online training method in TRL that supports vLLM and requires generating completions during training, such as DPO.

## Prerequisites

Ensure you have the necessary libraries installed. The key component is **TRL with vLLM support**, which enables high-throughput generation during online training.

```bash
pip install -U -q trl[vllm] peft math_verify trackio transformers
```

Authenticate with your Hugging Face account to save and share models directly.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Load and Prepare the Dataset

We'll use a mathematical reasoning dataset to train the model on complex, multi-step problem-solving.

### 1.1 Load the Dataset

We'll use the `AI-MO/NuminaMath-TIR` dataset, which contains mathematical problems, their solutions, and detailed reasoning steps.

```python
from datasets import load_dataset

dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'test[:10%]'])
```

### 1.2 Inspect the Dataset Structure

Let's examine the dataset format and a sample.

```python
print(train_dataset)
print(train_dataset[0])
```

### 1.3 Format the Dataset with a System Prompt

We'll format the dataset as a conversation, prompting the model to first think through the problem and then provide the final answer within specific tags. This structure encourages explicit reasoning.

```python
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

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

Check the formatted prompt:

```python
print(train_dataset[0]['prompt'])
```

### 1.4 Clean the Dataset

Remove unnecessary columns, keeping only the custom `prompt` and `solution` columns.

```python
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print(train_dataset)
```

## Step 2: Configure the Model and Training Setup

### 2.1 Load the Baseline Model

We'll use `Qwen/Qwen2-0.5B-Instruct` as our baseline policy model. It's lightweight and fits well within typical GPU memory constraints.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 2.2 Configure LoRA for Efficient Fine-Tuning

We'll use LoRA (Low-Rank Adaptation) to fine-tune the model efficiently by updating only a small subset of parameters.

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

### 2.3 Define Reward Functions

We'll implement two reward functions to guide the training:

1.  **Format Reward:** Ensures the model's output follows the required `<think>...</think><answer>...</answer>` structure.
2.  **Accuracy Reward:** Verifies if the model's solution matches the ground truth.

```python
import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

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

### 2.4 Configure GRPO Training Arguments

Now, configure the GRPO training parameters. We enable vLLM for high-throughput generation during training. Since this example uses a single GPU, we set `vllm_mode="colocate"` to share resources between the trainer and vLLM.

```python
from trl import GRPOConfig

output_dir = "Qwen2-0-5B-GRPO-vllm-trl"

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    gradient_accumulation_steps=16,
    num_train_epochs=1,

    # Data preprocessing parameters
    max_completion_length=512,  # Increased from default 256
    num_generations=8,
    max_prompt_length=512,

    # Reporting and saving
    report_to=["trackio"],
    project=output_dir,
    trackio_space_id=f"sergiopaniego/{output_dir}",
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,

    # vLLM configuration
    use_vllm=True,
    vllm_mode="colocate",
    # Optional vLLM parameters (commented with defaults):
    # vllm_model_impl='vllm',
    # vllm_enable_sleep_mode=False,
    # vllm_guided_decoding_regex=None,
    # vllm_server_base_url=None,
    # vllm_server_host='0.0.0.0',
    # vllm_server_port=8000,
    # vllm_server_timeout=240.0,
    # vllm_gpu_memory_utilization=0.3,
    # vllm_tensor_parallel_size=1
    # vllm_importance_sampling_correction=True,
    # vllm_importance_sampling_cap=2.0
)
```

## Step 3: Train the Model

### 3.1 Initialize the Trainer

Create the `GRPOTrainer`, passing the model, reward functions, training arguments, and dataset.

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)
```

### 3.2 Suppress Verbose Logs (Optional)

To keep the training output clean, you can suppress certain warnings and logs. Be cautious in production environments, as this might hide important information.

```python
import logging
import warnings
from transformers import logging as transformers_logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
transformers_logging.set_verbosity_warning()
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client.session")
```

### 3.3 Start Training

Begin the training process. You'll see logs indicating vLLM has been launched by TRL.

```python
trainer.train()
```

### 3.4 Save and Push the Model

Once training is complete, save the model locally and push it to the Hugging Face Hub.

```python
trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name=dataset_id)
```

## Performance Insights

With vLLM enabled, training on a single GPU achieves approximately **0.07 iterations/second**. Disabling vLLM (`use_vllm=False`) reduces performance to about **0.04 iterations/second**, demonstrating a **~75% speedup** from using vLLM in this basic configuration.

This setup provides a foundation for efficient online training. For further gains, you can experiment with parameters like `max_completion_length`, `num_generations`, and `max_prompt_length`, or scale to multi-GPU setups to fully leverage vLLM's high-throughput capabilities.