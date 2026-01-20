# Prompt Optimization for Language Models with DSPy GEPA

_Authored by: [Behrooz Azarkhalili](https://github.com/behroozazarkhalili)_

This notebook demonstrates how to use [DSPy](https://dspy.ai/)'s GEPA (Generalized Error-driven Prompt Augmentation) optimizer to improve language model performance on mathematical reasoning tasks. We'll work with the [NuminaMath-1.5 dataset](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) and show how GEPA can boost accuracy through automated prompt optimization.

**What you'll learn:**
- Setting up DSPy with language models ([OpenRouter](https://openrouter.ai/)) 
- Processing and filtering mathematical problem datasets
- Building a baseline Chain-of-Thought reasoning program
- Optimizing prompts with GEPA using error-driven feedback
- Evaluating improvements in model accuracy


GEPA works by analyzing errors, generating targeted feedback, and automatically refining prompts to address common failure patterns. This makes it particularly effective for complex reasoning tasks where prompt quality significantly impacts performance.

**Key Resources:**
- [DSPy Documentation](https://dspy.ai/learn/programming/)
- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)
- [GEPA Optimizer Guide](https://dspy.ai/api/optimizers/GEPA/)

## Installation and Setup

Install required dependencies and import libraries for DSPy, dataset processing, and model configuration.

**Installation Options:**
- **uv** - Fast Python package installer ([documentation](https://docs.astral.sh/uv/))
- **pip** - Traditional Python package manager

**Key Dependencies:**
- `dspy` - DSPy framework for language model programming
- `datasets` - Hugging Face datasets library for loading NuminaMath-1.5
- `python-dotenv` - Environment variable management for API keys


```python
# Install with uv (recommended - faster)
!uv pip install dspy datasets python-dotenv

# Alternative: Install with pip
# !pip install dspy datasets python-dotenv
```


```python
import dspy
from datasets import load_dataset
import os
```

## Language Model Configuration

Configure your language model - either local (Ollama) or cloud-based (OpenRouter) - for use with DSPy.


```python
from dotenv import load_dotenv
load_dotenv("../../.env")
```




    True



### Model Selection Rationale

**Main LM: `openrouter/openai/gpt-4.1-nano`**

*Primary Role:* High-volume inference during baseline evaluation and GEPA optimization iterations

*Key Selection Criteria:*
1. **Cost Efficiency** - $0.10/M input tokens, $0.40/M output tokens (~90% cheaper than GPT-4.1 or Claude)
2. **Low Latency** - Fastest GPT-4.1 variant, enables rapid iteration with 16-32 parallel threads
3. **Adequate Performance** - 60-65% baseline accuracy (MMLU: 80.1%, GPQA: 50.3%)
4. **Context Window** - 1M tokens for long chain-of-thought reasoning

---

**Reflection LM: `openrouter/qwen/qwen3-next-80b-a3b-thinking`**

*Primary Role:* Deep error analysis and prompt improvement during GEPA's reflection phase

*Key Selection Criteria:*
1. **Advanced Reasoning** - "Thinking" variant specialized for analytical reasoning and pattern identification
2. **Quality Over Speed** - ~16 reflection calls vs 2000+ inference calls, can afford slower, higher-quality model
3. **Context Handling** - 10M token context window for processing multiple training examples
4. **Cost Trade-off** - More expensive per token but negligible total cost due to low volume

**Architecture Philosophy:** Use a cheap, fast model for high-volume inference (99% of calls) and a smart, analytical model for low-volume reflection (1% of calls). This asymmetric design optimizes for both cost efficiency and learning quality.

### Understanding GEPA's Two-Model Architecture

GEPA's breakthrough innovation lies in its **dual-model approach** for reflective prompt optimization, which fundamentally differs from traditional single-model optimizers.

**Why Two Models?**

Traditional prompt optimizers rely on scalar metrics (accuracy scores) to guide improvements, essentially using trial-and-error without understanding *why* predictions fail. GEPA introduces a revolutionary approach by separating concerns:

**1. Student LM (Inference Model)**
- **Role**: Primary model that executes tasks and generates predictions
- **Characteristics**: Fast, cost-efficient, handles high-volume inference
- **Usage Pattern**: ~90-95% of all API calls during optimization
- **In This Notebook**: `openrouter/openai/gpt-4.1-nano`

**2. Reflection LM (Meta-Cognitive Model)**
- **Role**: Analyzes failures, identifies patterns, and generates prompt improvements
- **Characteristics**: Stronger reasoning, analytical depth, interpretability
- **Usage Pattern**: ~5-10% of API calls (only during reflection phases)
- **In This Notebook**: `openrouter/qwen/qwen3-next-80b-a3b-thinking`

**The Reflective Optimization Cycle:**

```
1. Student LM solves training problems → predictions
2. Metric provides rich textual feedback on failures
3. Reflection LM analyzes batches of failures → identifies patterns
4. Reflection LM generates improved prompt instructions
5. Student LM tests new prompts → validation
6. Repeat until convergence
```

**Research Foundation:**

This approach is detailed in the paper ["Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/abs/2507.19457), which demonstrates that reflective optimization with textual feedback outperforms reinforcement learning approaches on complex reasoning tasks.


```python
# ============================================
# OpenRouter Language Model Configuration
# ============================================
# Requires OPENROUTER_API_KEY environment variable
# Sign up at https://openrouter.ai/ to get your API key

# # Main LM for inference
open_router_lm = dspy.LM(
    'openrouter/openai/gpt-4.1-nano', 
    api_key=os.getenv('OPENROUTER_API_KEY'), 
    api_base='https://openrouter.ai/api/v1',
    max_tokens=65536,
    temperature=1.0
)

# # Reflection LM for GEPA optimization
reflection_lm = dspy.LM(
    'openrouter/qwen/qwen3-next-80b-a3b-thinking', 
    api_key=os.getenv('OPENROUTER_API_KEY'), 
    api_base='https://openrouter.ai/api/v1',
    max_tokens=65536,
    temperature=1.0
)

# Set OpenRouter as default LM
dspy.configure(lm=open_router_lm)

print("✅ OpenRouter LM configured successfully!")
print(f"Main model: openrouter/openai/gpt-4.1-nano")
print(f"Reflection model: openrouter/qwen/qwen3-next-80b-a3b-thinking")
```

    ✅ OpenRouter LM configured successfully!
    Main model: openrouter/openai/gpt-4.1-nano
    Reflection model: openrouter/qwen/qwen3-next-80b-a3b-thinking


## Dataset Preparation Functions

Helper functions to process the dataset, split it into train/val/test sets, and preview examples.


```python
def init_dataset(
    train_split_ratio: float = None, 
    test_split_ratio: float = None, 
    val_split_ratio: float = None, 
    sample_fraction: float = 1.0
) -> tuple[list, list, list]:
    """
    Initialize and split the NuminaMath-1.5 dataset into train/val/test sets.
    
    Loads the dataset, filters for numeric answers, converts to DSPy Examples,
    shuffles with fixed seed for reproducibility, and optionally samples a fraction.
    
    Args:
        train_split_ratio: Proportion for training (default: 0.5)
        test_split_ratio: Proportion for testing (default: 0.45)
        val_split_ratio: Proportion for validation (default: 0.05)
        sample_fraction: Fraction of dataset to use (default: 1.0 = full dataset)
    
    Returns:
        Tuple of (train_set, val_set, test_set) as lists of DSPy Examples
    
    Raises:
        AssertionError: If split ratios don't sum to 1.0
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


```python
train_set, val_set, test_set = init_dataset(sample_fraction=0.00025)

print(len(train_set), len(val_set), len(test_set))
```

    Total number of examples after filtering: 896215
    Sampled down to 224 examples.
    112 22 90



```python
print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])
```

    Problem:
    In the diagram, $AB = 15\text{ cm},$ $DC = 24\text{ cm},$ and $AD = 9\text{ cm}.$ What is the length of $AC,$ to the nearest tenth of a centimeter?
    
    [asy]
    draw((0,0)--(9,16)--(33,16)--(9,0)--cycle,black+linewidth(1));
    draw((9,16)--(9,0),black+linewidth(1));
    draw((0,0)--(33,16),black+linewidth(1));
    draw((9,0)--(9,0.5)--(8.5,0.5)--(8.5,0)--cycle,black+linewidth(1));
    draw((9,16)--(9.5,16)--(9.5,15.5)--(9,15.5)--cycle,black+linewidth(1));
    label("$A$",(0,0),NW);
    label("$B$",(9,16),NW);
    label("$C$",(33,16),E);
    label("$D$",(9,0),SE);
    label("15 cm",(0,0)--(9,16),NW);
    label("9 cm",(0,0)--(9,0),S);
    label("24 cm",(9,0)--(33,16),SE);
    [/asy]
    
    
    Solution:
    Extend $AD$ to point $E$ where it intersects the perpendicular from $C$ on $BC$'s extension.
    
    [asy]
    draw((0,0)--(9,16)--(33,16)--(9,0)--cycle,black+linewidth(1));
    draw((9,16)--(9,0),black+linewidth(1));
    draw((0,0)--(33,16),black+linewidth(1));
    draw((9,0)--(9,0.5)--(8.5,0.5)--(8.5,0)--cycle,black+linewidth(1));
    draw((9,16)--(9.5,16)--(9.5,15.5)--(9,15.5)--cycle,black+linewidth(1));
    label("$A$",(0,0),NW);
    label("$B$",(9,16),NW);
    label("$C$",(33,16),E);
    label("$D$",(9,0),SE);
    draw((9,0)--(33,0),black+linewidth(1)+dashed);
    draw((33,0)--(33,16),black+linewidth(1)+dashed);
    draw((33,0)--(33,0.5)--(32.5,0.5)--(32.5,0)--cycle,black+linewidth(1));
    label("$E$",(33,0),SE);
    label("18 cm",(9,0)--(33,0),S);
    label("16 cm",(33,0)--(33,16),E);
    [/asy]
    
    Using the Pythagorean theorem in $\triangle ADB$, calculate $BD^2 = BA^2 - AD^2 = 15^2 - 9^2 = 144$, so $BD = 12\text{ cm}$.
    
    In $\triangle DBC$, compute $BC^2 = DC^2 - BD^2 = 24^2 - 12^2 = 432$, thus $BC = 18\text{ cm}$.
    
    Recognize $BCED$ as a rectangle, hence $DE = BC = 18\text{ cm}$ and $CE = BD = 12\text{ cm}$.
    
    Examine $\triangle AEC$ with $AE = AD + DE = 9 + 18 = 27\text{ cm}$, then apply Pythagorean theorem:
    \[ AC^2 = AE^2 + CE^2 = 27^2 + 12^2 = 729 + 144 = 873 \]
    \[ AC = \sqrt{873} \approx \boxed{29.5\text{ cm}} \]
    
    
    Answer:
    29.5\text{ cm}



```python
print(test_set[0]['problem'])
print("\n\nAnswer:")
print(test_set[0]['answer'])
```

    a cistern is two - third full of water . pipe a can fill the remaining part in 12 minutes and pipe b in 8 minutes . once the cistern is emptied , how much time will they take to fill it together completely ?
    
    
    Answer:
    14.4


## Baseline Chain-of-Thought Program

Create a simple baseline using DSPy's Chain-of-Thought module to establish initial performance.


```python
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)
```

## Evaluation Metric

Define the evaluation metric to compare model predictions against ground truth answers.


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

## Baseline Evaluation

Evaluate the baseline Chain-of-Thought program to establish our starting accuracy before optimization.


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

    [Average Metric: 35.00 / 59 (59.3%), ..., Average Metric: 47.00 / 90 (52.2%)]

    2025/10/04 20:23:05 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.

    2025/10/04 20:23:25 INFO dspy.evaluate.evaluate: Average Metric: 47 / 90 (52.2%)


    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>problem</th>
      <th>solution</th>
      <th>example_answer</th>
      <th>reasoning</th>
      <th>pred_answer</th>
      <th>metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a cistern is two - third full of water . pipe a can fill the remai...</td>
      <td>First, let's find out how much time it would take for each pipe to...</td>
      <td>14.4</td>
      <td>The cistern is initially two-thirds full, so the remaining part to...</td>
      <td>4.8 minutes</td>
      <td></td>
