# Exploring New Features in Gemini 1.5 Pro & Flash 002

This guide explores the new capabilities introduced with the 002 versions of the Gemini 1.5 series models. You will learn how to use:
* **Candidate Count**: Generate multiple response candidates in a single request.
* **Presence & Frequency Penalties**: Control token repetition and encourage diversity in model outputs.
* **Response Log Probabilities**: Access the model's confidence scores for generated tokens.

## Prerequisites & Setup

First, ensure you have the correct version of the Google Generative AI SDK installed and your API key configured.

### 1. Install the SDK
Install a version of the SDK compatible with the 002 models.

```bash
pip install -q "google-generativeai>=0.8.2"
```

### 2. Configure Imports and API Key
Import the necessary libraries and configure your API key. If you are using Google Colab, you can store your key in the notebook secrets.

```python
import google.generativeai as genai

# For Google Colab, retrieve the API key from user data
from google.colab import userdata
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# For general display purposes
from IPython.display import display, Markdown, HTML
```

### 3. Check Available 002 Models
Let's verify which 002 models are available for use.

```python
for model in genai.list_models():
  if '002' in model.name:
    print(model.name)
```

You should see output similar to:
```
models/gemini-1.5-pro-002
models/gemini-1.5-flash-002
...
```

For this tutorial, we will use the Flash model and a simple test prompt.

```python
model_name = "models/gemini-1.5-flash-002"
test_prompt = "Why don't people have tails"
```

## A Quick Refresher on `generation_config`

Before diving into the new features, it's important to understand how to configure model generations. Parameters like `temperature` and `max_output_tokens` are set using a `generation_config`.

You can set this configuration in two ways:
1. When initializing the `GenerativeModel`.
2. When calling `generate_content` (or `chat.send_message`).

Any configuration passed to `generate_content` will override the settings on the model object. The configuration can be a Python dictionary or a `genai.GenerationConfig` object.

```python
# Example: Setting config during model initialization and overriding during generation
model = genai.GenerativeModel(model_name, generation_config={'temperature': 1.0})
response = model.generate_content('hello', generation_config=genai.GenerationConfig(max_output_tokens=5))
```

If you are ever unsure about the available parameters, you can inspect `genai.GenerationConfig`.

## 1. Using Candidate Count

With the 002 models, you can now request multiple candidate responses in a single API call by setting `candidate_count > 1`.

### Step 1: Initialize the Model and Set Configuration
First, create a model instance and define a generation config with `candidate_count`.

```python
model = genai.GenerativeModel(model_name)
generation_config = dict(candidate_count=2)
```

### Step 2: Generate Multiple Candidates
Pass your prompt and the configuration to the model.

```python
response = model.generate_content(test_prompt, generation_config=generation_config)
```

### Step 3: Accessing the Candidates
**Important Note**: The `.text` quick-accessor on the response only works when there is a single candidate (`candidate_count=1`). With multiple candidates, it will raise an error.

```python
try:
  response.text # This will fail
except ValueError as e:
  print(e)
# Output: Invalid operation: The `response.parts` quick accessor retrieves the parts for a single candidate...
```

To access the text from multiple candidates, you must iterate through the `response.candidates` list.

```python
for candidate in response.candidates:
  display(Markdown(candidate.content.parts[0].text))
  display(Markdown("-------------"))
```

This will display two distinct, complete answers to the prompt "Why don't people have tails". Each `Candidate` object in the response contains the full generated content, its finish reason, and other metadata like average log probabilities.

## 2. Applying Presence and Frequency Penalties

The 002 models introduce penalty arguments that influence the statistical likelihood of tokens appearing in the output, allowing you to control repetition and encourage diversity.

### Understanding Presence Penalty
The `presence_penalty` penalizes tokens that have already appeared in the output. A positive penalty encourages the model to use a more diverse vocabulary, while a negative penalty discourages diversity.

Let's create a helper function to measure the effect by calculating the fraction of unique words in multiple responses.

```python
from statistics import mean

def unique_words(prompt, generation_config, N=10):
  """Generates N responses and calculates the fraction of unique words in each."""
  vocab_fractions = []
  for n in range(N):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(contents=prompt, generation_config=generation_config)
    words = response.text.lower().split()
    # Calculate unique word ratio
    score = len(set(words)) / len(words)
    print(score)
    vocab_fractions.append(score)
  return vocab_fractions
```

### Step 1: Establish a Baseline
First, let's see the default vocabulary diversity without any penalty.

```python
prompt = 'Tell me a story'
baseline_scores = unique_words(prompt, generation_config={})
print(f"Baseline average unique word ratio: {mean(baseline_scores):.3f}")
```

### Step 2: Apply a Positive Presence Penalty
Now, apply a strong positive penalty to encourage more diverse word choice.

```python
positive_penalty_scores = unique_words(prompt, generation_config=dict(presence_penalty=1.999))
print(f"With positive penalty, average unique word ratio: {mean(positive_penalty_scores):.3f}")
```

### Step 3: Apply a Negative Presence Penalty
Finally, apply a negative penalty to see if it reduces vocabulary diversity.

```python
negative_penalty_scores = unique_words(prompt, generation_config=dict(presence_penalty=-1.999))
print(f"With negative penalty, average unique word ratio: {mean(negative_penalty_scores):.3f}")
```

You should observe that the `presence_penalty` has a measurable, though often subtle, effect on the output's lexical diversity.

### Understanding Frequency Penalty
The `frequency_penalty` is similar to `presence_penalty`, but the penalty is multiplied by the number of times a token has been used. This creates a much stronger effect, especially for repetitive tasks.

The easiest way to see this is to ask the model to repeat a word.

```python
model = genai.GenerativeModel(model_name)
response = model.generate_content(
    contents='please repeat "Cat" 50 times, 10 per line',
    generation_config=dict(frequency_penalty=1.999)
)
print(response.text)
```

You will see the model struggles to repeat "Cat" perfectly. It introduces variations like "cat", "CaT", or "CAT" because the heavy penalty on the frequently used "Cat" token forces it to choose alternatives.

**⚠️ Caution with Negative Frequency Penalty**
A negative frequency penalty makes a token *more* likely the more it is used. This positive feedback can cause the model to get stuck in a loop, repeating a common token until it hits the `max_output_tokens` limit and cannot generate the stop token.

```python
response = model.generate_content(
    prompt, # 'Tell me a story'
    generation_config=genai.GenerationConfig(
        max_output_tokens=400,
        frequency_penalty=-2.0)
)
# The output may become highly repetitive
display(Markdown(response.text))
print(f"Finish reason: {response.candidates[0].finish_reason}")
# The finish reason will likely be `MAX_TOKENS`, indicating the generation hit the token limit.
```

## Summary

You have now explored the key new features in the Gemini 1.5 Pro and Flash 002 models:
* **Candidate Count**: Generate and compare multiple alternative responses.
* **Presence Penalty**: Fine-tune vocabulary diversity in model outputs.
* **Frequency Penalty**: Apply a stronger, cumulative penalty to control token repetition, useful for preventing or inducing repetitive patterns.

These tools provide greater control over the creativity, variety, and structure of your model's generations. Remember to use negative frequency penalties with extreme caution to avoid infinite repetition loops.