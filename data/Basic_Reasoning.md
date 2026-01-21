# Guide: Basic Reasoning with the Gemini API

This guide demonstrates how to use the Gemini API to perform structured reasoning tasks, such as solving mathematical word problems and logical puzzles. You will learn how to configure a system prompt to guide the model through a clear, step-by-step reasoning process.

## Prerequisites

First, ensure you have the required Python package installed.

```bash
pip install -U "google-genai>=1.0.0"
```

## Step 1: Configure the API Client

Import the necessary modules and set up your Gemini API client. You will need a valid `GOOGLE_API_KEY`.

```python
from google import genai
from google.genai import types

# Replace with your actual API key
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 2: Select a Model

Choose a Gemini model for your task. This example uses a preview model, but you can select any available model that suits your needs.

```python
MODEL_ID = "gemini-3-flash-preview"  # Example model
```

## Step 3: Define a System Prompt for Structured Reasoning

To get consistent, step-by-step explanations, you can instruct the model with a specific system prompt. This prompt tells the model to act as a teacher and follow a clear problem-solving framework.

```python
system_prompt = """
You are a teacher solving mathematical and logical problems. Your task:
1. Summarize given conditions.
2. Identify the problem.
3. Provide a clear, step-by-step solution.
4. Provide an explanation for each step.

Ensure simplicity, clarity, and correctness in all steps of your explanation.
Each of your task should be done in order and separately.
"""

config = types.GenerateContentConfig(
    system_instruction=system_prompt
)
```

## Step 4: Solve a Logical Probability Problem

Now, let's apply this setup to a classic probability puzzle. We'll ask the model to determine whether a die is likely weighted after observing a roll of six.

```python
logical_problem = """
Assume a world where 1 in 5 dice are weighted and have 100% to roll a 6.
A person rolled a dice and rolled a 6.
Is it more likely that the die was weighted or not?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=logical_problem,
    config=config,
)

print(response.text)
```

**Expected Output (Summary):**
The model will break down the problem using Bayes' Theorem. It defines events, states known probabilities, calculates the total probability of rolling a six, and finally computes the conditional probability that the die was weighted given the observed six.

The conclusion will be similar to:
```
P(W|6) ≈ 0.6
P(F|6) ≈ 0.4
It is more likely (60% probability) that the die was weighted than it was fair (40% probability), given that a 6 was rolled.
```

## Step 5: Solve a Simple Geometry Problem

Next, test the same structured approach on a straightforward geometry calculation.

```python
math_problem = """
Given a triangle with base b=6 and height h=8, calculate its area
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=math_problem,
    config=config,
)

print(response.text)
```

**Expected Output (Summary):**
The model will follow the instructed steps: summarize the given base and height, identify the task as finding the area, apply the formula `Area = (1/2) * base * height`, and explain each calculation step.

The final answer will be:
```
The area of the triangle is 24.
```

## Conclusion

You have successfully used the Gemini API to perform basic reasoning tasks. By providing a clear system prompt, you can guide the model to produce structured, step-by-step solutions for both logical puzzles and mathematical problems.

## Next Steps

- Experiment with your own prompts and problems using the provided system prompt as a template.
- Explore other reasoning tasks in the [Gemini Cookbook repository](https://github.com/google-gemini/cookbook).
- Try modifying the `system_instruction` to tailor the model's response style for different domains, such as code debugging or data analysis.