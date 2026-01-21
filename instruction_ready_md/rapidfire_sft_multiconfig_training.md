# Fine-Tune a Customer Support Chatbot 20x Faster with RapidFire AI

This guide demonstrates how to fine-tune a Large Language Model (LLM) for a customer support chatbot using **Supervised Fine-Tuning (SFT)**. We'll leverage [RapidFire AI](https://github.com/RapidFireAI/rapidfireai) to concurrently train and compare four different model configurations on a single GPU, achieving **16-24x faster experimentation throughput** compared to sequential training.

## What You Will Build

You will create a chatbot capable of answering customer support queries in a helpful and friendly manner. You'll fine-tune the GPT-2 model on the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## Key Concepts & Benefits

*   **Concurrent Experimentation:** RapidFire AI uses chunk-based scheduling to interleave training across multiple configurations, providing comparative performance metrics early in the process.
*   **Parameter-Efficient Fine-Tuning (PEFT):** We'll use LoRA (Low-Rank Adaptation) adapters to efficiently update the base model with minimal new parameters.
*   **Interactive Control (IC Ops):** Manage training runs in real-time with stop, resume, clone-modify, and delete operations.
*   **Real-time Monitoring:** Track training progress and compare runs using TensorBoard.

## Prerequisites & Setup

This tutorial is designed to run in **Google Colab**. Ensure you have a Colab runtime with a GPU enabled (Runtime -> Change runtime type -> T4 GPU).

### Step 1: Install RapidFire AI

Run the following cell to install the `rapidfireai` package and initialize it. This process takes about two minutes.

```python
try:
    import rapidfireai
    print("✅ rapidfireai already installed")
except ImportError:
    %pip install rapidfireai
    !rapidfireai init
```

### Step 2: Start RapidFire AI Services

RapidFire AI runs background services for orchestration. Start them with the following code.

```python
import subprocess
from time import sleep
import socket

try:
  # Try to connect to the service ports to check if they're running
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
  sleep(30) # Wait for services to fully start
```

**Note:** You can also start the services by running `rapidfireai start` in a Colab terminal.

### Step 3: Configure TensorBoard for Logging

We'll use TensorBoard to visualize our experiments. Configure RapidFire AI to use it.

```python
import os

# Load the TensorBoard extension
%load_ext tensorboard

# Set the tracking backend to TensorBoard
os.environ['RF_TRACKING_BACKEND'] = 'tensorboard'
```

### Step 4: Import Required Libraries

Import the core RapidFire AI components and other necessary libraries.

```python
from rapidfireai import Experiment
from rapidfireai.fit.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
```

## Prepare the Training Data

We will use a subset of the Bitext Customer Support dataset for this demonstration, suitable for Colab's memory constraints.

### Step 5: Load and Sample the Dataset

```python
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Create smaller subsets for training and evaluation
train_dataset = dataset["train"].select(range(64))  # 64 training examples
eval_dataset = dataset["train"].select(range(50, 60))  # 10 evaluation examples

# Shuffle the datasets for randomness
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset = eval_dataset.shuffle(seed=42)
```

### Step 6: Format the Data for the Model

We need to format the instruction-response pairs into a text prompt suitable for GPT-2.

```python
def sample_formatting_function(example):
    """Format the dataset for GPT-2 while preserving original fields"""
    return {
        "text": f"Question: {example['instruction']}\nAnswer: {example['response']}",
        "instruction": example['instruction'],  # Keep original
        "response": example['response']  # Keep original
    }

# Apply the formatting function to both datasets
eval_dataset = eval_dataset.map(sample_formatting_function)
train_dataset = train_dataset.map(sample_formatting_function)
```

### Step 7: Define an Evaluation Metric

We'll compute ROUGE-L score to evaluate the quality of the model's generated responses.

```python
def sample_compute_metrics(eval_preds):
    """Lightweight metrics computation using ROUGE-L"""
    predictions, labels = eval_preds

    try:
        import evaluate
        rouge = evaluate.load("rouge")
        rouge_output = rouge.compute(
            predictions=predictions,
            references=labels,
            use_stemmer=True,
            rouge_types=["rougeL"]  # Only compute rougeL to save memory
        )
        return {"rougeL": round(rouge_output["rougeL"], 4)}
    except Exception as e:
        # Fallback if metrics fail
        print(f"Metrics computation failed: {e}")
        return {}
```

## Configure the RapidFire AI Experiment

### Step 8: Initialize the Experiment

Create an experiment with a unique name. This acts as a container for all your concurrent training runs.

```python
my_experiment = "tensorboard-demo-1"
experiment = Experiment(experiment_name=my_experiment)
```

### Step 9: Define the Model Configurations

We will explore a grid of 4 configurations, combining two LoRA adapter sizes with two different learning rate strategies.

```python
# Define two LoRA adapter configurations
peft_configs_lite = List([
    RFLoraConfig(
        r=8,  # Small rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # GPT-2's combined Q,K,V attention module
        bias="none"
    ),
    RFLoraConfig(
        r=32, # Large rank
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # Attention + output projection
        bias="none"
    )
])

# Define two main training configurations
config_set_lite = List([
    RFModelConfig(
        model_name="gpt2",
        peft_config=peft_configs_lite,
        training_args=RFSFTConfig(
            learning_rate=5e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,  # Effective batch size = 4
            max_steps=64,
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=4,
            per_device_eval_batch_size=2,
            fp16=True,
            gradient_checkpointing=True,  # Saves memory
            report_to="none",  # Disables Weights & Biases
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
            "pad_token_id": 50256,  # GPT-2's EOS token
        }
    ),
    RFModelConfig(
        model_name="gpt2",
        peft_config=peft_configs_lite,
        training_args=RFSFTConfig(
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=64,
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=4,
            per_device_eval_batch_size=2,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",
            warmup_steps=10,
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

### Step 10: Create a Model Factory Function

RapidFire AI uses a factory function to instantiate models on-demand during its chunk-based scheduling, preventing out-of-memory errors.

```python
def sample_create_model(model_config):
    """Function to create model object with GPT-2 adjustments"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    model_kwargs = model_config["model_kwargs"]

    # Load the causal language model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 specific tokenizer adjustments
    if "gpt2" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Better for autoregressive generation
        model.config.pad_token_id = model.config.eos_token_id

    return (model, tokenizer)
```

### Step 11: Define the Configuration Grid

Combine the configurations into a grid search object. This tells RapidFire AI to train all possible combinations (2 LoRA configs x 2 trainer configs = 4 total runs).

```python
config_group = RFGridSearch(
    configs=config_set_lite,
    trainer_type="SFT"
)
```

## Launch TensorBoard for Monitoring

Before starting training, launch TensorBoard to monitor all four runs in real-time.

```python
# Get the experiment path to find TensorBoard logs
from rapidfireai.fit.db.rf_db import RfDb
db = RfDb()
experiment_path = db.get_experiments_path(my_experiment)
tensorboard_log_dir = f"{experiment_path}/{my_experiment}/tensorboard_logs"

print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

# Launch TensorBoard
%tensorboard --logdir {tensorboard_log_dir}
```

A new panel will appear below the cell with interactive charts for loss, learning rate, and evaluation metrics across all runs.

## Run Concurrent Training

### Step 12: Start the Experiment

Kick off the concurrent training of all four configurations. RapidFire AI will interleave their training on data chunks.

```python
experiment.run_fit(
    config_group,
    sample_create_model,
    train_dataset,
    eval_dataset,
    num_chunks=4,  # Dataset is split into 4 chunks for interleaved scheduling
    seed=42
)
```

## Manage Runs with the Interactive Controller

After training starts, you can manage the runs dynamically.

### Step 13: Launch the Interactive Controller

The controller provides buttons to stop, resume, delete, or clone-modify any active run.

```python
from time import sleep
sleep(15) # Wait a moment for runs to initialize
from rapidfireai.fit.utils.interactive_controller import InteractiveController

controller = InteractiveController(dispatcher_url="http://127.0.0.1:8851")
```

An interactive widget will appear, listing all runs. You can:
*   **⏹️ Stop** a running configuration.
*   **▶️ Resume** a stopped run.
*   **🗑️ Delete** a run from the experiment.
*   **📋 Clone** a run and modify its configuration (e.g., change the learning rate) to start a new experiment, optionally warming it up from the parent's weights.
*   **🔄 Refresh** the status and metrics display.

## Conclusion

You have successfully set up and launched a concurrent fine-tuning experiment using RapidFire AI. By now, you should see four runs progressing in TensorBoard, allowing you to compare their performance in real-time.

**Key Takeaways:**
1.  **Faster Iteration:** You are training four configurations in roughly the same time it would take to train one sequentially.
2.  **Early Insights:** Metrics after the first data chunk give you early signals about which hyperparameters are promising.
3.  **Dynamic Control:** The Interactive Controller lets you adapt your experiment strategy on the fly, stopping poor performers and cloning promising ones.

To explore the full capabilities of RapidFire AI, including its advanced UI and production features, visit the [Install and Get Started](https://oss-docs.rapidfire.ai/en/latest/walkthrough.html) guide.