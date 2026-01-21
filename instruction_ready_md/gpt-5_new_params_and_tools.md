# GPT-5 New Parameters and Tools: A Practical Guide

This guide introduces the latest developer controls in the GPT-5 series, giving you greater control over model responses—from shaping output length and style to enforcing strict formatting. You'll learn how to use these features through practical, step-by-step examples.

## New Features Overview

| Feature | Overview | Values / Usage |
| :--- | :--- | :--- |
| **Verbosity Parameter** | Hints the model to be more or less expansive in its replies. Keep prompts stable and use the parameter instead of re-writing. | `low` (terse UX, minimal prose), `medium` (default, balanced detail), `high` (verbose, great for audits, teaching, or hand-offs). |
| **Freeform Function Calling** | Generate raw text payloads—anything from Python scripts to SQL queries—directly to your custom tool without JSON wrapping. Offers greater flexibility for external runtimes. | Use when structured JSON isn't needed and raw text is more natural for the target tool (e.g., code sandboxes, SQL databases, shell environments). |
| **Context-Free Grammar (CFG)** | A set of production rules defining valid strings in a language. Useful for constraining output to match the syntax of programming languages or custom formats. | Use as a contract to ensure the model emits only valid strings accepted by the grammar. |
| **Minimal Reasoning** | Runs GPT-5 with few or no reasoning tokens to minimize latency and speed time-to-first-token. Ideal for deterministic, lightweight tasks. | Set reasoning effort: `"minimal"`. Avoid for multi-step planning or tool-heavy workflows. |

**Supported Models:** `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
**Supported API Endpoints:** Responses API, Chat Completions API
**Recommendation:** Use the Responses API with GPT-5 series models for the best performance.

## Prerequisites

Begin by updating your OpenAI SDK to a version that supports the new GPT-5 parameters and tools. Ensure your `OPENAI_API_KEY` is set as an environment variable.

```bash
pip install --quiet --upgrade openai pandas
```

## 1. Using the Verbosity Parameter

The verbosity parameter lets you control the length and detail of the model's output without changing your prompt.

### 1.1 Basic Example: Generating a Poem

Let's see how verbosity affects a simple creative task.

```python
from openai import OpenAI
import pandas as pd

client = OpenAI()
question = "Write a poem about a boy and his first pet dog."

data = []

for verbosity in ["low", "medium", "high"]:
    response = client.responses.create(
        model="gpt-5-mini",
        input=question,
        text={"verbosity": verbosity}
    )

    # Extract text from the response
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content in item.content:
                if hasattr(content, "text"):
                    output_text += content.text

    usage = response.usage
    data.append({
        "Verbosity": verbosity,
        "Sample Output": output_text,
        "Output Tokens": usage.output_tokens
    })

# Create and display a comparison DataFrame
df = pd.DataFrame(data)
pd.set_option('display.max_colwidth', None)
print(df.to_string())
```

The output tokens scale roughly linearly with verbosity: **low (~560 tokens)** → **medium (~849 tokens)** → **high (~1288 tokens)**.

### 1.2 Practical Use Case: Code Generation

The verbosity parameter also influences the length, complexity, and accompanying explanations of generated code. Let's generate a Python program to sort an array of one million random numbers at different verbosity levels.

First, we'll define a helper function to make the calls.

```python
from openai import OpenAI

client = OpenAI()
prompt = "Output a Python program that sorts an array of 1000000 random numbers"

def ask_with_verbosity(verbosity: str, question: str):
    response = client.responses.create(
        model="gpt-5-mini",
        input=question,
        text={"verbosity": verbosity}
    )

    # Extract assistant's text output
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content in item.content:
                if hasattr(content, "text"):
                    output_text += content.text

    usage = response.usage

    print("--------------------------------")
    print(f"Verbosity: {verbosity}")
    print("Output:")
    print(output_text)
    print(f"Tokens => input: {usage.input_tokens} | output: {usage.output_tokens}")
```

#### Low Verbosity

```python
ask_with_verbosity("low", prompt)
```

**Output Summary:**
The model returns a minimal, functional script with no extra comments or structure. It simply generates the numbers, sorts them, and prints the timing and first/last 10 elements. Output token count is low (~575).

#### Medium Verbosity

```python
ask_with_verbosity("medium", prompt)
```

**Output Summary:**
The model adds explanatory comments, function structure, and reproducibility controls (like setting a random seed). It also includes notes about the sorting algorithm and suggests alternatives like NumPy for better performance. Output token count is medium (~943).

#### High Verbosity

```python
ask_with_verbosity("high", prompt)
```

**Output Summary:**
The model yields a comprehensive, production-ready script. It includes:
- Argument parsing for backend choice (pure Python vs. NumPy), array size, seed, and sample size.
- Separate implementations for both backends with error handling.
- Detailed timing for generation, sorting, and verification steps.
- Extensive usage examples and best-practice tips.
Output token count is high (~2381).

### 1.3 Key Takeaways

The new verbosity parameter reliably scales both the length and depth of the model’s output while preserving correctness and reasoning quality—**without changing the underlying prompt**.

- **Low verbosity**: Produces minimal, functional output. Ideal for concise answers in user interfaces.
- **Medium verbosity**: Adds explanatory context and structure. Perfect for general-purpose documentation and tutorials.
- **High verbosity**: Delivers comprehensive, production-ready output with extensive details and best practices. Excellent for audits, teaching, and complex hand-offs.

## 2. Free‑Form Function Calling

GPT‑5 can now send raw text payloads directly to your custom tool without wrapping the data in JSON, using the new `"type": "custom"` tool definition. This offers greater flexibility when interacting with external runtimes like code sandboxes, SQL databases, or shell environments.

**Note:** The custom tool type does **not** support parallel tool calling.

### 2.1 Quick Start Example: Compute the Area of a Circle

Let's create a simple example where the model uses a freeform tool call to execute Python code.

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input="Please use the code_exec tool to calculate the area of a circle with radius equal to the number of 'r's in strawberry",
    text={"format": {"type": "text"}},
    tools=[
        {
            "type": "custom",
            "name": "code_exec",
            "description": "Executes arbitrary python code",
        }
    ]
)

# Inspect the response
print(response.output)
```

The model emits a `tool call` containing raw Python code. In a real workflow, you would:
1.  Execute that code server‑side.
2.  Capture the printed result.
3.  Send the result back in a follow‑up `responses.create` call to continue the conversation.

### 2.2 Mini‑Benchmark: Sorting an Array in Three Languages

This example illustrates a more advanced use case. We'll ask GPT‑5 to:
1.  Generate Python, C++, and Java code that sorts a fixed array 10 times.
2.  Print only the time (in ms) taken for each iteration.
3.  Call all three functions.

First, we define the tools and a wrapper function for the API call.

```python
from openai import OpenAI
from typing import List, Optional

MODEL_NAME = "gpt-5"

# Define the custom tools for different languages
TOOLS = [
    {
        "type": "custom",
        "name": "code_exec_python",
        "description": "Executes python code",
    },
    {
        "type": "custom",
        "name": "code_exec_cpp",
        "description": "Executes c++ code",
    },
    {
        "type": "custom",
        "name": "code_exec_java",
        "description": "Executes java code",
    },
]

client = OpenAI()

def create_response(
    input_messages: List[dict],
    previous_response_id: Optional[str] = None,
):
    """Wrapper around client.responses.create."""
    # The function body would contain the API call logic.
    # This is a placeholder showing the intended structure.
    pass
```

In the next steps (which would follow in the complete tutorial), you would use this setup to:
1.  Craft an initial prompt instructing the model to generate and call the three sorting functions.
2.  Parse the model's response to extract the raw code for each language.
3.  Execute each code snippet in its respective sandboxed environment.
4.  Collect the timing results and feed them back to the model for analysis.

This pattern enables complex, multi-step workflows where the model orchestrates code generation and execution across different systems.

## Next Steps

You've learned how to control output detail with the **Verbosity Parameter** and how to integrate with external systems using **Free‑Form Function Calling**. To dive deeper, explore the official documentation for:
- **Context-Free Grammar (CFG)**: To enforce strict output formats.
- **Minimal Reasoning**: To optimize latency for simple, deterministic tasks.

Remember to use the Responses API with GPT-5 models to leverage the full performance of these new features.