# Gemini API Cookbook: Basic Code Generation

This guide demonstrates how to use the Gemini API for two common development tasks: explaining errors and generating new code. By the end, you'll be able to integrate AI-assisted coding into your workflow to save time on debugging and boilerplate creation.

## Prerequisites

### 1. Install the SDK
First, install the official Google Generative AI Python SDK.

```bash
pip install -U "google-genai>=1.0.0"
```

### 2. Configure Authentication
You need a Gemini API key. Store it securely and configure the client.

```python
from google import genai

# Replace with your actual API key
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
client = genai.Client(api_key=GOOGLE_API_KEY)
```

> **Note:** In a Google Colab environment, you can securely store your key as a Secret named `GOOGLE_API_KEY` and retrieve it with `from google.colab import userdata` and `userdata.get('GOOGLE_API_KEY')`.

### 3. Select a Model
Choose a model suitable for code tasks. The `gemini-2.5-flash` models offer a good balance of speed and capability for this use case.

```python
MODEL_ID = "gemini-2.5-flash"
```

---

## Part 1: AI-Powered Error Explanation

When you encounter a cryptic error, you can ask the model to explain it and suggest a fix. For deterministic, accurate explanations, set the temperature to `0`.

### Step 1: Define the System Prompt
Create a system instruction that defines the model's role for this task.

```python
from google.genai import types

error_handling_system_prompt = """
Your task is to explain exactly why this error occurred and how to fix it.
"""

error_handling_model_config = types.GenerateContentConfig(
    temperature=0,  # Ensures deterministic, factual output
    system_instruction=error_handling_system_prompt
)
```

### Step 2: Craft Your Query
Formulate a prompt that includes the actual error message you received.

```python
error_message = """
    1 my_list = [1,2,3]
  ----> 2 print(my_list[3])

  IndexError: list index out of range
"""

error_prompt = f"""
You've encountered the following error message:
Error Message: {error_message}
"""
```

### Step 3: Generate the Explanation
Send the prompt to the model using the configured client.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=error_prompt,
    config=error_handling_model_config
)

print(response.text)
```

**Example Output:**
The model will return a detailed breakdown:
*   **What the error means:** "You're trying to access a list element using an index that doesn't exist."
*   **Why it happened:** "Python list indices start at 0. Your list `[1, 2, 3]` has valid indices 0, 1, and 2. Index `3` is out of range."
*   **How to fix it:** Provides corrected code snippets, such as using `my_list[2]` to get the last element or `my_list[-1]` with negative indexing.

---

## Part 2: Generating Code Snippets

You can also use the model as a coding assistant to generate functional code from a description.

### Step 1: Define the Coding Assistant Role
Configure the model with a system prompt tailored for code generation.

```python
code_generation_system_prompt = """
You are a coding assistant. Your task is to generate a code snippet that
accomplishes a specific goal. The code snippet must be concise, efficient,
and well-commented for clarity. Consider any constraints or requirements
provided for the task.

If the task does not specify a programming language, default to Python.
"""

code_generation_model_config = types.GenerateContentConfig(
    temperature=0,  # Low temperature for reliable, functional code
    system_instruction=code_generation_system_prompt
)
```

### Step 2: Describe Your Task
Write a clear, instructional prompt describing the code you need.

```python
code_generation_prompt = """
Create a countdown timer that ticks down every second and prints
"Time is up!" after 20 seconds
"""
```

### Step 3: Generate and Review the Code
Request the code snippet from the model.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=code_generation_prompt,
    config=code_generation_model_config
)

print(response.text)
```

**Example Output:**
The model returns a complete, ready-to-use Python function.

```python
import time

def countdown_timer(seconds):
    """
    Counts down from a specified number of seconds and prints "Time is up!" when finished.

    Args:
        seconds (int): The number of seconds to count down from.
    """
    for i in range(seconds, 0, -1):  # Iterate from seconds down to 1
        print(f"{i} seconds remaining...")
        time.sleep(1)  # Pause for 1 second
    print("Time is up!")

if __name__ == "__main__":
    countdown_timer(20)  # Start a 20-second countdown
```

### Step 4: Execute the Generated Code (Optional)
You can automatically extract and run the Python code from the model's response.

```python
import re

# Extract the code block from the markdown response
matchFound = re.search(r"```python\n(.*?)```", response.text, re.DOTALL)
if matchFound:
    code = matchFound.group(1)
    exec(code)
```

When executed, this will print:
```
20 seconds remaining...
19 seconds remaining...
...
1 seconds remaining...
Time is up!
```

---

## Summary and Next Steps

You've successfully used the Gemini API to:
1.  **Debug errors** by getting plain-English explanations and fixes.
2.  **Generate code** from a simple natural language description.

**To explore further:**
*   Experiment with different **model IDs** (like `gemini-2.5-pro` for more complex reasoning).
*   Adjust the **system prompts** to tailor the assistant for specific frameworks or coding styles.
*   Try more advanced tasks like code optimization, translating code between languages, or writing unit tests.

Remember to always review and test AI-generated code before using it in production.