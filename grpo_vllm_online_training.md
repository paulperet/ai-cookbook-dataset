# Efficient Online Training with GRPO and vLLM in TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

Online training methods, such as **Group Relative Policy Optimization (GRPO)** and **Direct Preference Optimization (DPO)**, require the model to **generate outputs in real time** during training. This "online" aspect often becomes a critical bottleneck, as generating completions is both **compute- and memory-intensive**, especially for large language models (LLMs).

Without optimization, running inference during training can be **slow and memory-heavy**, limiting both efficiency and scalability. This is particularly noticeable when hardware resources are constrained, such as in Colab with a single GPU.

This notebook demonstrates how to **overcome the online generation bottleneck** by combining **vLLM**, a high-throughput, low-latency inference engine built on **PagedAttention**, with **TRL**. On a single GPU, TRL and vLLM can share resources efficiently, enabling faster training even with limited hardware. On larger setups, such as multi-GPU or multi-node environments, vLLM can run as a separate process on dedicated GPUs while TRL handles training on others, allowing seamless scaling without impacting generation speed.

Although we focus on GRPO here, this setup is compatible with **any online training method in TRL with vLLM support that requires generating completions during training**, such as DPO. With minimal adjustments, the workflow can be adapted to different online optimization algorithms and hardware configurations while taking full advantage of efficient inference.

By using vLLM alongside TRL, we can directly observe measurable gains in **training efficiency**, with faster generation, reduced memory usage, and the ability to scale across multiple GPUs or nodes when needed.

The diagram below illustrates the overall training workflow and highlights where **vLLM** (blue box) and **TRL** (pink box) fit into the process:

## 1. Install Dependencies

First, let's install the essential libraries required for fine-tuning.
The important highlight here is **TRL with vLLM support**, which enables **high-throughput, low-latency generation** during online training, removing the common bottleneck in completion generation.

```python
!pip install -U -q trl[vllm] peft math_verify trackio transformers

# Tested with trl==0.23.1, peft==0.17.1, math_verify==0.8.0, vllm==0.11.0, trackio==0.5.2 transformers==4.57.0
```

Authenticate with your Hugging Face account to save and share your model directly from this notebook 🗝️.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Load Dataset 📁

These models excel at tasks that require **complex, multi-step reasoning**.
A prime example is **mathematical problem-solving**, where step-by-step thinking is essential to arrive at the correct answer.

For this project, we'll use the [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) dataset.  
This **reasoning-focused dataset** contains mathematical problems, their final solutions, and, most importantly, **detailed reasoning steps** that explain how to move from the problem statement to the solution.

```python
from datasets import load_dataset

dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'test[:10%]'])
```

Let's check the structure of the dataset

```python
print(train_dataset)
```

Let's check one sample:

```python
print(train_dataset[0])
```

In the **DeepSeek-R1** training procedure (where GRPO was first introduced, as described in the [previous notebook](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl)), a specific **system prompt** was used to guide the model in generating both **reasoning steps** and the **final answer** in a structured format.

We'll adopt the same approach here, formatting our dataset so that each example represents a **conversation between a User and an Assistant**. The Assistant is prompted to first think through the problem before providing the final solution.

The system prompt used is:

```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: prompt. Assistant:
```

This conversational structure ensures that the model **explicitly demonstrates its reasoning** before giving the answer, which is crucial for enhancing multi-step reasoning skills in mathematical problem-solving tasks.

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

Let's take a look at an example:

```python
print(train_dataset[0]['prompt'])
```

We'll remove the `messages` and `problem` columns, as we only need the custom `prompt` column and `solution` to verify the generated answer.

```python
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print(train_dataset)
```

## 3. Post-Training the Base Model Using GRPO + vLLM ⚡

A key challenge in online methods like GRPO is that the model must generate completions during training, which can quickly become a bottleneck. By integrating **vLLM**, we enable **high-throughput, low-latency generation** via its [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) mechanism. This not only speeds up the post-training loop but also improves memory efficiency, making large-scale reasoning tasks more practical.

TRL supports online training with vLLM in two different modes:

- **`colocate`**: The trainer process and the vLLM process share the same GPU resources. This is the setup used in this notebook, since Colab provides only a single GPU.
- **`server`**: The trainer and vLLM run on separate GPUs. This mode is ideal for multi-GPU setups, where TRL can use some GPUs for training while vLLM uses others, communicating via HTTP.

These modes provide flexibility to efficiently leverage available hardware while benefiting from vLLM's fast generation.

### 3.1 Loading the Baseline Model

We'll start by loading [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) as our baseline (the **Policy Model** in the diagram above).  
With just **0.5B parameters**, this model is lightweight and fits comfortably within typical GPU memory.  

- For improved performance, you may consider a [larger alternative model](https://x.com/jiayi_pirate/status/1882839487417561307).  
- We intentionally avoid the newer **Qwen2.5** or **Qwen3** series, since they are already optimized for reasoning/maths tasks, as also [highlighted by other developers](https://thinkingmachines.ai/blog/lora/#reinforcement-learning).

Later in the workflow, **vLLM will reuse this same model for generation**. Importantly, we **don't need to initialize vLLM here**—TRL will handle initialization automatically once the training loop begins, thanks to **colocate mode** (explained earlier).  

We'll see how this comes into play in the next steps.

```python
import torch
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 3.2 Configuring LoRA ⚙️

Next, we'll configure **LoRA** (Low-Rank Adaptation) for model training.  
LoRA allows us to **fine-tune the model efficiently** by updating a small set of parameters instead of the full model, resulting in **faster training** and **lower GPU memory usage**.

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

### 3.3 Loading Reward Functions

For the reward component of the system, we can use either pretrained reward models or reward functions defined directly in code. For training, the DeepSeek-R1 authors used an accuracy-based reward model evaluates whether the response is correct, alongside a format-based reward that ensures the model places its reasoning process between `<think> </think>` tags. You can find more details [here](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py). We can simply define and implement these reward functions as generic Python functions.

In this case, we will utilize these reward functions:

1. **Format Enforcement:** Ensures that the generation follows a specific format using `<think> </think> <answer> </answer>` tags for reasoning.

```python
import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]
```

2. **Solution Accuracy:** Verifies whether the solution to the problem is correct.

```python
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
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

### 3.4 Configuring GRPO Training Parameters

Next, we'll configure the training parameters for GRPO. Key parameters to experiment with are `max_completion_length`, `num_generations`, and `max_prompt_length` (see the diagram at the beginning for details on each).  

To keep things simple, we'll start with **just one training epoch**. We've doubled the `max_completion_length` so the model can generate slightly longer answers than the default in the `GRPOConfig` of 256 tokens. In practice, we recommend setting `num_generations` to 8 or more, as this has virtually no impact on GPU memory. The same principle applies to other parameters—careful experimentation and fine-tuning are key to identifying the most effective configuration for your task. In the next section, we provide a table showing training speeds for different parameter settings.

We'll also enable **vLLM** for generation during training. This is done by setting `use_vllm=True`, which instructs TRL to automatically launch and manage vLLM once the training loop begins.

Since this notebook runs on **a single GPU**, we configure **`colocate` mode** (via the `vllm_mode` parameter), so both the trainer and vLLM share the same GPU resources. In multi-GPU setups, you can instead run vLLM in a separate process, dedicating specific GPUs to each and letting them communicate via HTTP—unlocking even greater efficiency.

For more advanced configurations, check out the [official vLLM integration guide](https://huggingface.co/docs/trl/main/en/vllm_integration). In multi-GPU environments, you can also launch vLLM with the `trl vllm-serve` tool to further maximize throughput and performance.

```python
from trl import GRPOConfig

output_dir = "Qwen2-0-5B-GRPO-vllm-trl"

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    gradient_accumulation_steps=16,
    num_train_epochs=1,

    # Parameters that control de data preprocessing
    max_completion_length=512,  # default: 256
    num_generations=8,  # default: 8
    max_prompt_length=512,  # default: 512

    # Parameters related to reporting and saving
    report_to=["trackio"],
    project=output_dir, # For trackio
    trackio_space_id=f"sergiopaniego/{output_dir}", # For trackio
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,

    # Configure vLLM
    use_vllm=True,
    vllm_mode="colocate",
    # Some more params you can configure for vLLM with their defaults
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

We're reporting the training results to [trackio](https://huggingface.co/docs/trackio/en/index). To keep track of metrics and monitor them live during training, we can set up a **Hugging Face Space**, where the tracking will be continuously updated. We added `project` and `trackio_space_id` in `GRPOConfig` to configure it.

### 3.5 Training the Model 🏃

Next, we'll configure the trainer and begin training the model.

For this setup, we pass the two reward functions we defined earlier to the trainer to guide the learning process.

Below is a diagram illustrating the training procedure we'll be reproducing, adapted from the [Open-R1 project](https://github.com/huggingface/open-r1).

Finally, let’s configure the `GRPOTrainer`.

If you look closely at the output, you'll see details about the launch of vLLM. Thanks to TRL, integrating vLLM is straightforward, with minimal friction—allowing you to easily take advantage of high-throughput generation during online training.

For a deeper understanding of the benefits, we recommend comparing this notebook with the previous GRPO recipe without vLLM.

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)
```

We'll suppress certain warnings and logs to keep the output clean during training. Since training involves loops, some logs can appear repeatedly and may not be helpful for our example. In a real setting, be careful when suppressing logs, as important information could be hidden. If you want to see the full trace, you can ignore this cell.

```python
import logging
import warnings
from transformers import logging as transformers_logging

logging.basicConfig(level=logging.WARNING) # Set global logging level to WARNING
logging.getLogger("vllm").setLevel(logging.WARNING)  # Silence INFO logs from vLLM
transformers_logging.set_verbosity_warning() # Set Transformers logging to WARNING

# Ignore specific Python warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client.session")
```

Time to train the model! 🎉

```python
trainer.train()
```

Let's save the results 💾

```python
trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name=dataset_id)
```

In the HF Space, you can review the training results tracked by `trackio`. The metrics look very promising!

The setup shown here runs on a single GPU, yet we can already see how vLLM boosts training efficiency. With vLLM enabled, training reaches **0.07 it/s**, whereas disabling it (`use_vllm=False`) drops performance to **0.04 it/s**—an immediate **~75% speedup** even in this basic configuration.  

And this is just the beginning: we haven't yet explored more optimal setups. For further efficiency gains, you can experiment with training parameters like `max_completion_length`, `num_generations`, or `max_prompt_length`, and scale across multiple GPUs to fully leverage vLLM's high-throughput generation.

## 4. Evaluating Different Training Configurations

After training a model efficiently with a single configuration, it's insightful to explore other possible configurations to understand how training performance changes when using vLLM versus not using it. The table below shows various configurations along