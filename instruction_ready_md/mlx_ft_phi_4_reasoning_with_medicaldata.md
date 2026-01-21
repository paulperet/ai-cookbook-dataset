# Fine-Tuning Phi-4-mini-reasoning for Medical Reasoning with Apple MLX

This guide walks you through fine-tuning the `Phi-4-mini-reasoning` model on a medical reasoning dataset using Apple's MLX framework. The goal is to enhance the model's ability to reason about medical events by training it on the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset.

> **Note:** This sample is designed to run on Apple Silicon devices with at least 64GB of memory.

## Prerequisites

First, install the required Python packages.

```bash
pip install -U mlx-lm datasets
```

## Step 1: Prepare the Training Data

We'll use the Hugging Face `datasets` library to load and format our data. The dataset will be split into training and validation sets, then saved in the format expected by the MLX training script.

### 1.1 Define the Prompt Template and Formatting Function

Create a prompt template that structures each example for the model. The template follows a specific conversation format with `<think>` and `<assistant>` tags.

```python
from datasets import load_dataset

prompt_template = """<|user|>{}<|end|><|assistant|><think>{}</think>{}<|end|>"""

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = prompt_template.format(input, cot, output) + "<|endoftext|>"
        texts.append(text)
    return {"text": texts}
```

### 1.2 Load and Split the Dataset

Load the medical reasoning dataset and split it into an 80/20 train/validation split.

```python
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", trust_remote_code=True)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=200)

train_dataset = split_dataset['train']
validation_dataset = split_dataset['test']
```

### 1.3 Apply Formatting and Save the Datasets

Apply the formatting function to both splits, remove the original columns, and save the results as JSONL files in a `data` directory. The MLX framework expects training data as `train.jsonl` and validation data as `valid.jsonl`.

```python
# Format and save the training set
train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])
train_dataset.to_json("./data/train.jsonl")

# Format and save the validation set
validation_dataset = validation_dataset.map(formatting_prompts_func, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])
validation_dataset.to_json("./data/valid.jsonl")
```

## Step 2: Fine-Tune the Model with LoRA

Now, we'll fine-tune the `Phi-4-mini-reasoning` model using Low-Rank Adaptation (LoRA). This technique is efficient and allows us to train a small subset of parameters.

Run the following command from your terminal. This will train the model for 100 iterations using the data in the `./data` directory.

```bash
python -m mlx_lm.lora --model ./phi-4-mini-reasoning --train --data ./data --iters 100
```

**Key outputs during training:**
*   The script reports the percentage of trainable parameters (a very small fraction, thanks to LoRA).
*   Training progress is logged, showing iteration number, training/validation loss, learning rate, and performance metrics.
*   The final LoRA adapter weights are saved to the `./adapters` directory.

## Step 3: Test the Fine-Tuned Model

Finally, let's generate a response with our fine-tuned model to see its improved medical reasoning capabilities.

We'll provide a medical case as a prompt and instruct the model to generate a response. The `--adapter-path` flag points to our trained LoRA weights.

```bash
python -m mlx_lm.generate --model ./phi-4-mini-reasoning --adapter-path ./adapters --max-tokens 4096 --prompt "A 54-year-old construction worker with a long history of smoking presents with swelling in his upper extremity and face, along with dilated veins in this region. After conducting a CT scan and venogram of the neck, what is the most likely diagnosis for the cause of these symptoms?" --extra-eos-token "<|end|>"
```

**Example Model Output:**
The model will generate a reasoned response. It first produces a `<think>` section showing its internal reasoning process, followed by a final, concise diagnosis.

```
<think>Okay, let's see. The patient is a 54-year-old construction worker with a long history of smoking... [Detailed reasoning omitted for brevity] ...So, the diagnosis is a venous thrombosis in the subclavian vein.</think>The patient’s symptoms—swelling in the upper extremity and face along with dilated veins in the neck—are consistent with increased venous pressure and venous obstruction. Given his history of smoking, which is a known risk factor for thrombosis, the most likely diagnosis is a venous thrombosis in the subclavian vein.
```

The script also reports performance statistics like tokens generated per second and peak memory usage.

## Summary

You have successfully fine-tuned the `Phi-4-mini-reasoning` model for enhanced medical reasoning. The process involved:
1.  Preparing a specialized medical dataset in the correct format.
2.  Efficiently training the model using LoRA via the MLX framework.
3.  Testing the model's new capabilities by generating a diagnosis for a complex medical case.

The fine-tuned model now demonstrates structured reasoning (`<think>`) before delivering a final answer, a behavior learned from the `medical-o1-reasoning-SFT` dataset.