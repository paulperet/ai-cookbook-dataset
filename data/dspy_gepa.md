# Optimizing Language Model Prompts with DSPy GEPA

_Authored by: [Behrooz Azarkhalili](https://github.com/behroozazarkhalili)_

This guide demonstrates how to use DSPy's GEPA (Generalized Error-driven Prompt Augmentation) optimizer to improve language model performance on mathematical reasoning tasks. We'll use the [NuminaMath-1.5 dataset](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) to show how automated prompt optimization can boost accuracy.

**What you'll learn:**
- Setting up DSPy with language models via OpenRouter
- Processing and filtering mathematical problem datasets
- Building a baseline Chain-of-Thought reasoning program
- Optimizing prompts with GEPA using error-driven feedback
- Evaluating improvements in model accuracy

GEPA works by analyzing errors, generating targeted feedback, and automatically refining prompts to address common failure patterns. This makes it particularly effective for complex reasoning tasks where prompt quality significantly impacts performance.

## Prerequisites

Install the required dependencies.

```bash
pip install dspy datasets python-dotenv
```

## 1. Setup and Imports

Import the necessary libraries.

```python
import dspy
from datasets import load_dataset
import os
from dotenv import load_dotenv
```

## 2. Configure the Language Models

Load your environment variables and configure the language models. GEPA uses a dual-model architecture for optimal performance.

```python
# Load API key from .env file
load_dotenv("../../.env")

# Configure the main inference model (Student LM)
open_router_lm = dspy.LM(
    'openrouter/openai/gpt-4.1-nano', 
    api_key=os.getenv('OPENROUTER_API_KEY'), 
    api_base='https://openrouter.ai/api/v1',
    max_tokens=65536,
    temperature=1.0
)

# Configure the reflection model for GEPA optimization
reflection_lm = dspy.LM(
    'openrouter/qwen/qwen3-next-80b-a3b-thinking', 
    api_key=os.getenv('OPENROUTER_API_KEY'), 
    api_base='https://openrouter.ai/api/v1',
    max_tokens=65536,
    temperature=1.0
)

# Set the main model as the default
dspy.configure(lm=open_router_lm)

print("✅ OpenRouter LM configured successfully!")
print(f"Main model: openrouter/openai/gpt-4.1-nano")
print(f"Reflection model: openrouter/qwen/qwen3-next-80b-a3b-thinking")
```

**Model Selection Rationale:**

- **Main LM (`gpt-4.1-nano`)**: Used for high-volume inference during evaluation and optimization. It's cost-efficient and fast.
- **Reflection LM (`qwen3-next-80b-a3b-thinking`)**: Used for deep error analysis during GEPA's reflection phase. It's slower but provides superior reasoning for identifying failure patterns.

This asymmetric design optimizes for both cost efficiency and learning quality.

## 3. Prepare the Dataset

Define a helper function to load and split the NuminaMath-1.5 dataset.

```python
def init_dataset(
    train_split_ratio: float = None, 
    test_split_ratio: float = None, 
    val_split_ratio: float = None, 
    sample_fraction: float = 1.0
) -> tuple[list, list, list]:
    """
    Initialize and split the NuminaMath-1.5 dataset into train/val/test sets.
    
    Args:
        train_split_ratio: Proportion for training (default: 0.5)
        test_split_ratio: Proportion for testing (default: 0.45)
        val_split_ratio: Proportion for validation (default: 0.05)
        sample_fraction: Fraction of dataset to use (default: 1.0 = full dataset)
    
    Returns:
        Tuple of (train_set, val_set, test_set) as lists of DSPy Examples
    """
    # Set default split ratios
    if train_split_ratio is None:
        train_split_ratio = 0.5
    if test_split_ratio is None:
        test_split_ratio = 0.4
    if val_split_ratio is None:
        val_split_ratio = 0.1
    
    # Validate split ratios sum to 1.0
    assert (train_split_ratio + test_split_ratio + val_split_ratio) == 1.0, "Ratios must sum to 1.0"

    # Load dataset from Hugging Face Hub
    train_split = load_dataset("AI-MO/NuminaMath-1.5")['train']
    
    # Convert to DSPy Examples with input/output fields
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")  # Mark 'problem' as input field
        for x in train_split
    ]
    
    # Shuffle with fixed seed for reproducibility
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)
    print(f"Total number of examples after filtering: {tot_num}")

    # Apply sampling if requested
    if sample_fraction < 1.0:
        sample_num = int(tot_num * sample_fraction)
        train_split = train_split[:sample_num]
        tot_num = sample_num
        print(f"Sampled down to {sample_num} examples.")
    
    # Split into train/val/test based on ratios
    train_end = int(train_split_ratio * tot_num)
    val_end = int((train_split_ratio + val_split_ratio) * tot_num)
    
    train_set = train_split[:train_end]
    val_set = train_split[train_end:val_end]
    test_set = train_split[val_end:]

    return train_set, val_set, test_set
```

Now, initialize the dataset with a small sample for demonstration.

```python
train_set, val_set, test_set = init_dataset(sample_fraction=0.00025)

print(f"Train set: {len(train_set)} examples")
print(f"Validation set: {len(val_set)} examples")
print(f"Test set: {len(test_set)} examples")
```

Let's examine a sample problem to understand the dataset structure.

```python
print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])
```

## 4. Create a Baseline Chain-of-Thought Program

First, define a signature that specifies the input and output fields.

```python
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()
```

Now, create a baseline program using DSPy's ChainOfThought module.

```python
program = dspy.ChainOfThought(GenerateResponse)
```

## 5. Define an Evaluation Metric

We need a metric to compare model predictions against ground truth answers. This function parses numeric answers from the text.

```python
def parse_integer_answer(answer):
    try:
        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError, TypeError):
        answer = 0

    return answer

def metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

## 6. Evaluate the Baseline

Establish a baseline accuracy before optimization using the test set.

```python
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=16,
    display_table=True,
    display_progress=True
)

evaluate(program)
```

The baseline accuracy will be displayed. For this sample, you should see an accuracy around **52.2%**. This is our starting point before applying GEPA optimization.

## Summary

You've now set up the environment, prepared the dataset, created a baseline Chain-of-Thought program, and established an evaluation metric. In the next section, you'll apply GEPA optimization to improve the model's performance through automated prompt refinement.