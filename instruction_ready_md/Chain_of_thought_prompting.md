# Guide: Implementing Chain-of-Thought Prompting with the Gemini API

This guide demonstrates how to use chain-of-thought prompting to improve the reasoning capabilities of large language models (LLMs) when solving logic and arithmetic problems. You'll learn how to structure prompts to guide the model through a step-by-step reasoning process, leading to more accurate and reliable answers.

## Prerequisites

### 1. Install Required Libraries
First, install the Google Generative AI Python SDK:

```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Configure Your API Key
To authenticate with the Gemini API, you need a valid API key. Store your key in a secure location and configure the client:

```python
from google.colab import userdata
from google import genai

# Retrieve your API key from Colab Secrets
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

> **Note:** If you're not using Google Colab, you can set the `GOOGLE_API_KEY` environment variable directly instead of using `userdata.get()`.

### 3. Select a Model
Choose a Gemini model for your experiments. The following models support chain-of-thought reasoning:

```python
MODEL_ID = "gemini-3-flash-preview"  # You can change this to other available models
```

Available model options include:
- `gemini-2.5-flash-lite`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gemini-2.5-flash-preview`
- `gemini-3-pro-preview`

## Understanding the Problem: Direct Prompting Limitations

LLMs sometimes provide unsatisfactory answers when asked to solve complex reasoning problems directly. Let's examine this issue with a classic rate problem.

### Step 1: Test Without Chain-of-Thought

First, we'll ask the model to solve a problem with the instruction to "Return the answer immediately," which discourages step-by-step reasoning:

```python
from IPython.display import Markdown

prompt = """
5 people can create 5 donuts every 5 minutes. How much time would it take
25 people to make 100 donuts? Return the answer immediately.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)
Markdown(response.text)
```

The model might return a simple but potentially incorrect answer like "5 minutes" without showing its work. This demonstrates the need for a more structured approach.

## Implementing Chain-of-Thought Prompting

Chain-of-thought prompting encourages the model to break down complex problems into smaller, manageable steps before arriving at a final answer.

### Step 2: Create a Chain-of-Thought Prompt Template

The key to effective chain-of-thought prompting is providing an example that demonstrates the desired reasoning process. Here's a template that includes a solved example followed by the target problem:

```python
prompt = """
Question: 11 factories can make 22 cars per hour. How much time would it take 22 factories to make 88 cars?
Answer: A factory can make 22/11=2 cars per hour. 22 factories can make 22*2=44 cars per hour. Making 88 cars would take 88/44=2 hours. The answer is 2 hours.

Question: 5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
Answer:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)
Markdown(response.text)
```

### Step 3: Analyze the Chain-of-Thought Response

When you run the above prompt, the model should produce a detailed, step-by-step solution:

```
Here's how to solve the donut problem:

*   **Donuts per person per minute:** If 5 people make 5 donuts in 5 minutes, then one person makes one donut in 5 minutes (5 donuts / 5 people = 1 donut per person).
*   **Donuts by 25 people in 5 minutes:** 25 people can make 25 donuts every 5 minutes (25 people * 1 donut/person = 25 donuts).
*   **How many 5-minute intervals?** To make 100 donuts, it would take four 5-minute intervals (100 donuts / 25 donuts per interval = 4 intervals).
*   **Total Time:** 4 intervals * 5 minutes/interval = 20 minutes.

**Answer:** It would take 25 people 20 minutes to make 100 donuts.
```

Notice how the model:
1. Breaks the problem into logical steps
2. Shows calculations at each stage
3. Clearly explains the reasoning process
4. Arrives at the correct answer (20 minutes)

## Key Takeaways

1. **Chain-of-thought prompting** significantly improves the model's ability to solve complex reasoning problems by encouraging step-by-step thinking.

2. **Provide examples** of the desired reasoning format in your prompt. The model learns from the structure you show it.

3. **Be explicit** about wanting step-by-step reasoning. While some models might use chain-of-thought spontaneously, it's more reliable to explicitly guide them.

4. **Test different formulations** of your chain-of-thought examples to find what works best for your specific use case.

## Next Steps

Now that you understand chain-of-thought prompting:

1. **Experiment with different problem types** - Try applying this technique to logic puzzles, word problems, or mathematical proofs.

2. **Explore other prompting techniques** - Combine chain-of-thought with few-shot prompting, where you provide multiple examples of problems and solutions.

3. **Test with your own data** - Apply chain-of-thought prompting to domain-specific problems relevant to your work.

4. **Compare model performance** - Try the same chain-of-thought prompts with different Gemini models to see which produces the most reliable reasoning.

By mastering chain-of-thought prompting, you'll be able to extract more accurate and explainable reasoning from LLMs, making them more valuable tools for complex problem-solving tasks.