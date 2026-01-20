Join Discord if you need help + ⭐ _Star us on [GitHub](https://github.com/RapidFireAI/rapidfireai)_ ⭐
👉 **Note:** This Colab notebook illustrates simplified usage of `rapidfireai`. For the full RapidFire AI experience with advanced experiment manager, UI, and production features, see our [Install and Get Started](https://oss-docs.rapidfire.ai/en/latest/walkthrough.html) guide.
🎬 Watch our [intro video](https://youtu.be/nPMBfZWqPWI) to get started!

⚠️ **Important:** Avoid leaving this Colab tab idle for more than 5 minutes—Colab may disconnect. To stay connected, periodically refresh TensorBoard or run a cell.

## 20x Faster TRL Fine-tuning with RapidFire AI

_Authored by: [RapidFire AI Team](https://github.com/RapidFireAI)_

This cookbook demonstrates how to fine-tune LLMs using **Supervised Fine-Tuning (SFT)** with [RapidFire AI](https://github.com/RapidFireAI/rapidfireai), enabling you to train and compare multiple configurations concurrently—even on a single GPU. We'll build a customer support chatbot and explore how RapidFire AI's chunk-based scheduling delivers **16-24× faster experimentation throughput**.

**What You'll Learn:**

- **Concurrent LLM Experimentation**: How to define and run multiple SFT experiments concurrently
- **LoRA Fine-tuning**: Using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters of different capacities
- **Experiment Tracking**: TensorBoard logging and real-time dashboard monitoring
- **Interactive Control Operations (IC Ops)**: Using Stop, Resume, Clone-Modify, and Delete to manage runs mid-training

**Key Benefits of RapidFire AI:**

- ⚡ **16-24× Speedup**: Compare multiple configurations in the time it takes to run one sequentially
- 🎯 **Early Signals**: Get comparative metrics after the first data chunk instead of waiting for full training
- 🔧 **Drop-in Integration**: Uses familiar TRL/Transformers APIs with minimal code changes
- 📊 **Real-time Monitoring and Control**: Live dashboard with IC Ops (Stop, Resume, Clone-Modify, and Delete) on active runs

## What We're Building

In this tutorial, we'll fine-tune a **customer support chatbot** that can answer user queries in a helpful and friendly manner. We'll use the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), which contains instruction-response pairs covering common customer support scenarios—each example includes a user question and an ideal assistant response.

### Our Approach

We'll use **Supervised Fine-Tuning (SFT)** with **LoRA (Low-Rank Adaptation)** to efficiently adapt a pre-trained LLM (GPT-2) for customer support tasks. To find the best hyperparameters, we'll compare **4 configurations** simultaneously:

- **2 LoRA adapter sizes**: Small (rank 8) vs. Large (rank 32)
- **2 learning rates**: 5e-4 vs. 2e-4

RapidFire AI's chunk-based scheduling trains all configurations concurrently—processing the dataset in chunks and letting every run train on each chunk before moving to the next. This gives you comparative metrics early, so you can identify the best configuration without waiting for all training to complete.

The figure below illustrates this concept with 3 configurations (M1, M2, M3). Sequential training completes one configuration entirely before starting the next. RapidFire AI interleaves all configurations, training each on one data chunk before rotating to the next. The bottom row shows how IC Ops let you adapt mid-training—stopping underperformers and cloning promising runs. Our tutorial uses 4 configurations, but the scheduling principle is the same.

*Sequential vs. RapidFire AI on a single GPU with chunk-based scheduling and IC Ops.*

## Install RapidFire AI Package and Setup
### Option 1: Run Locally (or on a VM)
For the full RapidFire AI experience—advanced experiment management, UI, and production features—we recommend installing the package on a machine you control (for example, a VM or your local machine) rather than Google Colab. See our [Install and Get Started](https://oss-docs.rapidfire.ai/en/latest/walkthrough.html) guide.

### Option 2: Run in Google Colab
For simplicity, you can run this notebook on Google Colab. This notebook is configured to run end-to-end on Colab with no local installation required.

```python
try:
    import rapidfireai
    print("✅ rapidfireai already installed")
except ImportError:
    %pip install rapidfireai  # Takes ~1 min
    !rapidfireai init # Takes ~1 min
```

## Start RapidFire Services

Start the RapidFire AI services:

```python
import subprocess
from time import sleep
import socket
try:
  s = [socket.socket(socket.AF_INET, socket.SOCK_STREAM), socket.socket(socket.AF_INET, socket.SOCK_STREAM), socket.socket(socket.AF_INET, socket.SOCK_STREAM)]
  s[0].connect(("127.0.0.1", 8851))
  s[1].connect(("127.0.0.1", 8852))
  s[2].connect(("127.0.0.1", 8853))
  s[0].close()
  s[1].close()
  s[2].close()
  print("RapidFire Services are running")
except OSError as error:
  print("RapidFire Services are not running, launching now...")
  subprocess.Popen(["rapidfireai", "start"])
  sleep(30)
```

**Note:** You can also run `rapidfireai start` from the Colab **terminal** instead of the cell above.

## Configure RapidFire to Use TensorBoard

```python
import os

# Load TensorBoard extension
%load_ext tensorboard

# Configure RapidFire to use TensorBoard
os.environ['RF_TRACKING_BACKEND'] = 'tensorboard'
# TensorBoard log directory will be auto-created in experiment path
```

## Import RapidFire Components

```python
from rapidfireai import Experiment
from rapidfireai.fit.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
# If you get "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" from Colab, just rerun this cell
```

## Load and Prepare the Dataset

We'll use the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), which contains instruction-response pairs for training customer support chatbots.

```python
from datasets import load_dataset

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# REDUCED dataset for memory constraints in Colab
train_dataset = dataset["train"].select(range(64))  # Reduced from 128
eval_dataset = dataset["train"].select(range(50, 60))  # 10 examples
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset = eval_dataset.shuffle(seed=42)
```

## Define Data Processing Function

We'll format the data as Q&A pairs for GPT-2:

```python
def sample_formatting_function(example):
    """Format the dataset for GPT-2 while preserving original fields"""
    return {
        "text": f"Question: {example['instruction']}\nAnswer: {example['response']}",
        "instruction": example['instruction'],  # Keep original
        "response": example['response']  # Keep original
    }

# Apply formatting to datasets
eval_dataset = eval_dataset.map(sample_formatting_function)
train_dataset = train_dataset.map(sample_formatting_function)
```

## Define Metrics Function

We'll use a lightweight metrics computation with just ROUGE-L to save memory:

```python
def sample_compute_metrics(eval_preds):
    """Lightweight metrics computation"""
    predictions, labels = eval_preds

    try:
        import evaluate

        # Only compute ROUGE-L (skip BLEU to save memory)
        rouge = evaluate.load("rouge")
        rouge_output = rouge.compute(
            predictions=predictions,
            references=labels,
            use_stemmer=True,
            rouge_types=["rougeL"]  # Only compute rougeL
        )

        return {
            "rougeL": round(rouge_output["rougeL"], 4),
        }
    except Exception as e:
        # Fallback if metrics fail
        print(f"Metrics computation failed: {e}")
        return {}
```

## Initialize Experiment

```python
# Create experiment with unique name
my_experiment = "tensorboard-demo-1"
experiment = Experiment(experiment_name=my_experiment)
```

## Get TensorBoard Log Directory

The TensorBoard logs are stored in the experiment directory. Let's get the path:

```python
# Get experiment path
from rapidfireai.fit.db.rf_db import RfDb

db = RfDb()
experiment_path = db.get_experiments_path(my_experiment)
tensorboard_log_dir = f"{experiment_path}/{my_experiment}/tensorboard_logs"

print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
```

## Define Model Configurations

We'll use `RFGridSearch` to create a grid of all possible combinations from our configurations. This tutorial uses GPT-2 (124M parameters), which fits comfortably within Colab's memory constraints.

Our config group combines **2 LoRA adapters** (small: `r=8` targeting `c_attn`; large: `r=32` targeting `c_attn` + `c_proj`) with **2 training strategies** (Config A: `lr=5e-4`, linear scheduler; Config B: `lr=2e-4`, cosine scheduler with warmup). This produces the following **4 concurrent runs**:

| Run | Base Model | Learning Rate | Scheduler | LoRA Rank | Target Modules |
|-----|------------|---------------|-----------|-----------|----------------|
| 1   | gpt2       | 5e-4          | linear    | 8         | c_attn         |
| 2   | gpt2       | 5e-4          | linear    | 32        | c_attn, c_proj |
| 3   | gpt2       | 2e-4          | cosine    | 8         | c_attn         |
| 4   | gpt2       | 2e-4          | cosine    | 32        | c_attn, c_proj |

RapidFire AI trains all 4 configurations concurrently using chunk-based scheduling, giving you comparative metrics early so you can identify the best hyperparameters faster.

```python
# GPT-2 specific LoRA configs - different module names!
peft_configs_lite = List([
    RFLoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # GPT-2 combines Q,K,V in c_attn
        bias="none"
    ),
    RFLoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # c_attn (QKV) + c_proj (output)
        bias="none"
    )
])

# 2 configs with GPT-2
config_set_lite = List([
    RFModelConfig(
        model_name="gpt2",  # Only 124M params
        peft_config=peft_configs_lite,
        training_args=RFSFTConfig(
            learning_rate=5e-4,  # Low lr for more stability
            lr_scheduler_type="linear",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,  # Effective bs = 4
            max_steps=64, # Raise this to see more learning
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=4,
            per_device_eval_batch_size=2,
            fp16=True,
            gradient_checkpointing=True,  # Save memory
            report_to="none",  # Disables wandb
        ),
        model_type="causal_lm",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16",  # Explicit fp16
            "use_cache": False
        },
        formatting_func=sample_formatting_function,
        compute_metrics=sample_compute_metrics,
        generation_config={
            "max_new_tokens": 128,  # Reduced from 256
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "pad_token_id": 50256,  # GPT-2's EOS token
        }
    ),
    RFModelConfig(
        model_name="gpt2",
        peft_config=peft_configs_lite,
        training_args=RFSFTConfig(
            learning_rate=2e-4,  # Even more conservative
            lr_scheduler_type="cosine",  # Try cosine schedule
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=64,  # Increase to observe more learning behavior
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=4,
            per_device_eval_batch_size=2,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",  # Disables wandb
            warmup_steps=10,  # Add warmup for stability
        ),
        model_type="causal_lm",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16",
            "use_cache": False
        },
        formatting_func=sample_formatting_function,
        compute_metrics=sample_compute_metrics,
        generation_config={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "pad_token_id": 50256,
        }
    )
])
```

## Define the Model Factory Function

RapidFire AI uses a **factory function** to create model instances on-demand. Instead of loading all 4 models into memory at once (which would likely cause out-of-memory errors), RapidFire calls this function each time it needs a model during chunk-based scheduling. The function takes a configuration dictionary and returns a `(model, tokenizer)` tuple.

```python
def sample_create_model(model_config):
    """Function to create model object with GPT-2 adjustments"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    model_kwargs = model_config["model_kwargs"]

    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        # Default to causal LM
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 specific: Set pad token (GPT-2 doesn't have one by default)
    if "gpt2" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # GPT-2 works better with left padding
        model.config.pad_token_id = model.config.eos_token_id

    return (model, tokenizer)
```

```python
# Simple grid search across all config combinations: 4 total (2 LoRA configs × 2 trainer configs)
config_group = RFGridSearch(
    configs=config_set_lite,
    trainer_type="SFT"
)
```

## Monitor Training Loss and Evaluation Metrics

We'll use [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize training progress across all 4 configurations. TensorBoard provides interactive plots for loss curves, learning rates, and evaluation metrics—making it easy to compare which hyperparameter combinations perform best.

Run the cell below before starting training to see metrics update in real-time.

```python
%tensorboard --logdir {tensorboard_log_dir}
```

## Run Training and Evaluation

We'll now train all 4 configurations concurrently and evaluate them on the validation set. RapidFire AI handles the scheduling, rotating between configurations after each data chunk so you get comparative metrics early.

The `experiment.run_fit()` function orchestrates this process:

- **`config_group`** — The grid of configurations to train (our 4 combinations)
- **`sample_create_model`** — Factory function that creates model/tokenizer instances
- **`train_dataset`** / **`eval_dataset`** — Training and evaluation data
- **`num_chunks`** — Number of data chunks for interleaved scheduling (higher = more frequent rotation between configs)
- **`seed`** — Random seed for reproducibility

```python
# Launch train and validation for all configs in the config_group with swap granularity of 4 chunks for hyperparallel execution
experiment.run_fit(
    config_group,
    sample_create_model,
    train_dataset,
    eval_dataset,
    num_chunks=4,
    seed=42
)
```

## Launch Interactive Run Controller

RapidFire AI provides an Interactive Controller that lets you manage executing runs dynamically in real-time from the notebook:

- ⏹️ **Stop**: Gracefully stop a running config
- ▶️ **Resume**: Resume a stopped run
- 🗑️ **Delete**: Remove a run from this experiment
- 📋 **Clone**: Create a new run by editing the config dictionary of a parent run to try new knob values; optional warm start of parameters
- 🔄 **Refresh**: Update run status and metrics

The Controller uses ipywidgets and is compatible with both Colab (ipywidgets 7.x) and Jupyter (ipywidgets 8.x).

```python
# Create Interactive Controller
sleep(15)
from rapidfireai.fit.utils.interactive_controller import InteractiveController

controller = InteractiveController(dispatcher_url="http://127.0.0.1:8851")
