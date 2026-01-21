# Guide: Using Gemini's Thinking Mode for Complex Reasoning

This guide demonstrates how to use Gemini's thinking mode to solve complex reasoning tasks. You'll learn how to configure the model to expose its internal reasoning process and analyze the token usage.

## Prerequisites

First, set up your environment and authenticate.

### 1. Install Required Libraries
Ensure you have the `requests` library installed.

```bash
pip install requests
```

### 2. Set Up Authentication
Store your Google API key in a secure location. In this example, we'll retrieve it from a Colab secret named `GOOGLE_API_KEY`.

```python
from google.colab import userdata
import json
import requests

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Configure the Model Endpoint
Define the model you want to use and construct the API URL.

```python
MODEL_ID = "gemini-3-flash-preview"  # You can change this to other thinking models
url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GOOGLE_API_KEY}"
```

## Example 1: Brain Teaser with Thinking Mode

Let's start with a classic brain teaser to see the model's reasoning in action.

### Step 1: Define the Prompt
We'll ask the model to solve a 20-questions style puzzle.

```python
prompt = """
    You are playing the 20 question game. You know that what you are looking for
    is an aquatic mammal that doesn't live in the sea, and that's smaller than a
    cat. What could that be and how could you make sure?
"""
```

### Step 2: Configure Thinking Mode
To enable thinking mode, include a `thinkingConfig` object in your request. Setting `includeThoughts` to `True` tells the model to expose its reasoning process.

```python
data = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ],
    "generationConfig": {
        "thinkingConfig": {
            "includeThoughts": True,
        }
    }
}
```

### Step 3: Make the API Request
Send the request to the Gemini API.

```python
response = requests.post(
    url,
    headers={'Content-Type': 'application/json'},
    data=json.dumps(data)
).json()
```

### Step 4: Parse and Display the Response
The response contains both the model's internal thoughts and its final answer.

```python
print("THOUGHTS:")
print(response['candidates'][0]['content']['parts'][0]['text'])
print()
print("OUTPUT:")
print(response['candidates'][0]['content']['parts'][1]['text'])
```

**Expected Output Structure:**
The model will first show its reasoning process (thoughts), then provide a structured final answer listing possible animals and a strategy for confirmation.

### Step 5: Analyze Token Usage
After making the API call, inspect the `usageMetadata` to understand how tokens were allocated between thinking and final output.

```python
print("Prompt tokens:", response["usageMetadata"]["promptTokenCount"])
print("Thoughts tokens:", response["usageMetadata"]["thoughtsTokenCount"])
print("Output tokens:", response["usageMetadata"]["candidatesTokenCount"])
print("Total tokens:", response["usageMetadata"]["totalTokenCount"])
```

You'll typically see that the thinking process consumes a significant portion of the token budget, which reflects the model's extensive internal reasoning.

## Controlling the Thinking Budget

You can fine-tune how much the model "thinks" before answering.

### Disabling Thinking Steps
For non-pro models, you can disable thinking entirely by setting `thinkingBudget` to `0`. This forces the model to generate a final answer without an explicit reasoning step.

```python
if "-pro" not in MODEL_ID:
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 0,
            }
        }
    }

    response = requests.post(
        url,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    ).json()

    print(response['candidates'][0]['content']['parts'][0]['text'])
else:
    print("You can't disable thinking for pro models.")
```

**Note:** When thinking is disabled, the model's answer may be less thorough or may miss nuanced possibilities (like the platypus in our example).

## Key Takeaways

1. **Thinking Mode Enhances Reasoning:** By exposing the model's internal thoughts, you gain insight into how it approaches complex problems.
2. **Token Allocation:** The thinking process consumes tokens separately from the final output. Monitor `thoughtsTokenCount` to understand the cost of reasoning.
3. **Budget Control:** Use `thinkingBudget` to control how much computation the model dedicates to reasoning (where supported).
4. **Model Compatibility:** Not all models support disabling thinking. Pro models typically require thinking mode for optimal performance.

This approach is particularly valuable for:
- Debugging model reasoning
- Educational applications where you want to show the problem-solving process
- Complex tasks where understanding the model's chain of thought is as important as the final answer

Experiment with different `thinkingBudget` values and prompts to see how the model's reasoning and answers change accordingly.