# Gemini API: Zero-Shot Prompting Guide

Zero-shot prompting allows you to use Gemini models to answer queries without providing any examples or additional context. This technique is ideal for straightforward tasks that don't require specific formatting or complex reasoning.

## Prerequisites

### 1. Install the Required Library
First, install the Google Generative AI Python SDK.

```bash
pip install -U "google-genai>=1.0.0"
```

### 2. Configure Your API Key
You need a Google AI API key. Store it securely and load it into your environment. This example shows how to retrieve it from an environment variable.

```python
from google import genai
import os

# Retrieve your API key from an environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 3. Select a Model
Choose a Gemini model for your tasks. For this guide, we'll use `gemini-2.5-flash` for its speed and cost-effectiveness.

```python
MODEL_ID = "gemini-2.5-flash"
```

## Zero-Shot Prompting Examples

Now, let's explore several practical examples. In each case, you simply provide the task instruction directly to the model.

### Example 1: Sorting Items
**Task:** Sort a list of animals from largest to smallest.

```python
prompt = """
    Sort following animals from biggest to smallest:
    fish, elephant, dog
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
```
1. Elephant
2. Dog
3. Fish
```

### Example 2: Sentiment Analysis
**Task:** Classify the sentiment of a product review.

```python
prompt = """
    Classify sentiment of review as positive, negative or neutral:
    I go to this restaurant every week, I love it so much.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
```
Positive
```

### Example 3: Information Extraction
**Task:** Extract capital cities mentioned in a travel narrative.

```python
prompt = """
    Extract capital cities from the text:
    During the summer I visited many countries in Europe. First I visited Italy, specifically Sicily and Rome.
    Then I visited Cologne in Germany and the trip ended in Berlin.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
```
Rome and Berlin
```

### Example 4: Code Debugging
**Task:** Identify and fix a type error in a Python function.

```python
prompt = """
    Find and fix the error in this Python code:
    def add_numbers(a, b):
        return a + b
    print(add_numbers(5, "10"))
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
The model will explain the `TypeError` (adding an integer to a string) and provide corrected code, such as:
```python
def add_numbers(a, b):
    return a + b
print(add_numbers(5, 10))
```

### Example 5: Solving Math Problems
**Task:** Calculate average speed from distance and time.

```python
prompt = """
    Solve this math problem:
    A train travels 120 km in 2 hours. What is its average speed?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
The model will show the calculation: `120 km / 2 hours = 60 km/h`.

### Example 6: Named Entity Recognition (NER)
**Task:** Identify people, places, and countries in a news sentence.

```python
prompt = """
    Identify the names of people, places, and countries in this text:
    Emmanuel Macron, the president of France, announced a AI partnership in collaboration with the United Arab Emirates.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
```
People: Emmanuel Macron
Countries: France, United Arab Emirates
```

### Example 7: Grammar Correction
**Task:** Correct grammatical errors in a sentence.

```python
prompt = """
    Correct the grammar in this sentence:
    She don't like playing football but she enjoy to watch it.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
The model will provide the corrected sentence: `She doesn't like playing football, but she enjoys watching it.` and may include an explanation of the subject-verb agreement errors.

## Key Takeaways

Zero-shot prompting with Gemini is powerful for:
- **Simple Classification:** Sentiment, categories, or entity types.
- **Basic Transformations:** Sorting, extraction, or correction.
- **Straightforward Q&A:** Factual questions or simple calculations.

The model performs these tasks based on its pre-trained knowledge without needing examples. For more complex tasks requiring specific formats, reasoning steps, or niche domains, consider techniques like **few-shot prompting** or **chain-of-thought**.

## Next Steps

- Experiment with your own data and prompts.
- Explore advanced prompting techniques like [few-shot prompting](https://github.com/google-gemini/cookbook/blob/main/examples/prompting/Few_shot_prompting.ipynb) for tasks requiring examples.
- Try different Gemini models (e.g., `gemini-2.5-pro`) for more complex reasoning.