# Guide: Using Gemini's Thinking Capabilities for Complex Reasoning Tasks

## Overview

Gemini 2.5 and 3 models feature built-in thinking processes that enable stronger reasoning capabilities. These models expose their reasoning steps, allowing you to understand how they reach conclusions. This guide demonstrates how to leverage these thinking capabilities through practical examples.

## Prerequisites

### 1. Install the SDK
```bash
pip install -U -q 'google-genai>=1.51.0'
```

### 2. Set Up Your API Key
Store your API key in a Colab Secret named `GOOGLE_API_KEY`:

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the Client
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Configure Model Selection
```python
MODEL_ID = "gemini-3-flash-preview"  # Choose from available thinking models
```

### 5. Import Required Libraries
```python
import json
from PIL import Image
from IPython.display import display, Markdown
```

## Understanding Thinking Models

Thinking models are optimized for complex tasks requiring multiple rounds of strategizing and iterative solving. Key concepts:

- **Thinking Budget**: Controls maximum tokens for thinking (0-24576 for 2.5 Flash)
- **Thinking Levels**: Gemini 3 introduces simplified thinking budget management
- **Adaptive Thinking**: Default mode where the model dynamically adjusts thinking based on task complexity

## Step 1: Using Adaptive Thinking

Start with a reasoning task to see the model's thinking process in action. The model will identify a platypus based on specific characteristics.

```python
prompt = """
    You are playing the 20 question game. You know that what you are looking for
    is a aquatic mammal that doesn't live in the sea, is venomous and that's
    smaller than a cat. What could that be and how could you make sure?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

Markdown(response.text)
```

**Expected Output:**
The model should identify a platypus and explain its reasoning process, including:
- Confirmation of aquatic mammal status
- Freshwater habitat verification
- Venomous spur characteristic
- Size comparison to cats

### Analyzing Token Usage

Examine how tokens were allocated between thinking and response generation:

```python
print("Prompt tokens:", response.usage_metadata.prompt_token_count)
print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
print("Total tokens:", response.usage_metadata.total_token_count)
```

**Sample Output:**
```
Prompt tokens: 59
Thoughts tokens: 1451
Output tokens: 815
Total tokens: 2325
```

## Step 2: Disabling Thinking Steps

For non-pro models, you can disable thinking by setting `thinking_budget=0`. This demonstrates how thinking affects response quality.

```python
if "-pro" not in MODEL_ID:
    prompt = """
        You are playing the 20 question game. You know that what you are looking for
        is a aquatic mammal that doesn't live in the sea, is venomous and that's
        smaller than a cat. What could that be and how could you make sure?
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    Markdown(response.text)
else:
    print("You can't disable thinking for pro models.")
```

**Note:** Without thinking, the model may provide less accurate or different answers. In this case, it might suggest a freshwater water shrew instead of a platypus.

### Verify Token Allocation Without Thinking

```python
print("Prompt tokens:", response.usage_metadata.prompt_token_count)
print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
print("Total tokens:", response.usage_metadata.total_token_count)
```

**Sample Output:**
```
Prompt tokens: 59
Thoughts tokens: None
Output tokens: 688
Total tokens: 747
```

## Step 3: Solving a Physics Problem

Test the model's reasoning on a structural engineering calculation. First, try without thinking enabled:

```python
if "-pro" not in MODEL_ID:
    prompt = """
        A cantilever beam of length L=3m has a rectangular cross-section (width b=0.1m, height h=0.2m) and is made of steel (E=200 GPa).
        It is subjected to a uniformly distributed load w=5 kN/m along its entire length and a point load P=10 kN at its free end.
        Calculate the maximum bending stress (σ_max).
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    Markdown(response.text)
else:
    print("You can't disable thinking for pro models.")
```

**Expected Output:** The model should provide a step-by-step calculation including:
1. Understanding of cantilever beam concepts
2. Calculation of moment of inertia
3. Determination of maximum bending moment
4. Application of the flexure formula
5. Final maximum bending stress value

## Key Takeaways

1. **Thinking models excel at complex reasoning**: They break down problems systematically before providing answers
2. **Token allocation is transparent**: You can see exactly how many tokens were used for thinking vs. response
3. **Thinking affects response quality**: Disabling thinking can lead to different (often less accurate) answers
4. **Model selection matters**: Pro models always use thinking; Flash models allow thinking budget control

## Next Steps

For more advanced usage:
- Experiment with different thinking budgets (e.g., 1000, 5000, 10000 tokens)
- Try the same prompts with thinking enabled vs. disabled
- Test with various problem types (mathematical, logical, creative)
- Explore Gemini 3's thinking levels for simplified budget management

Remember that thinking capabilities are particularly valuable for:
- Complex problem-solving
- Multi-step reasoning tasks
- Tasks requiring verification or explanation
- Educational or tutorial content generation