# GPT-5 New Params and Tools

We’re introducing new developer controls in the GPT-5 series that give you greater control over model responses—from shaping output length and style to enforcing strict formatting. Below is a quick overview of the latest features:

| #  | Feature | Overview | Values / Usage |
|----|---------|----------|----------------|
| 1. | **Verbosity Parameter** | Lets you hint the model to be more or less expansive in its replies. Keep prompts stable and use the parameter instead of re-writing. | • **low** → terse UX, minimal prose.<br>• **medium** *(default)* → balanced detail.<br>• **high** → verbose, great for audits, teaching, or hand-offs. |
| 2. | **Freeform Function Calling** | Generate raw text payloads—anything from Python scripts to SQL queries—directly to your custom tool without JSON wrapping. Offers greater flexibility for external runtimes like:<br>• Code sandboxes (Python, C++, Java, …)<br>• SQL databases<br>• Shell environments<br>• Config generators | Use when structured JSON isn’t needed and raw text is more natural for the target tool. |
| 3. | **Context-Free Grammar (CFG)** | A set of production rules defining valid strings in a language. Each rule rewrites a non-terminal into terminals and/or other non-terminals, independent of surrounding context. Useful for constraining output to match the syntax of programming languages or custom formats in OpenAI tools. | Use as a contract to ensure the model emits only valid strings accepted by the grammar. |
| 4. | **Minimal Reasoning** | Runs GPT-5 with few or no reasoning tokens to minimize latency and speed time-to-first-token. Ideal for deterministic, lightweight tasks (extraction, formatting, short rewrites, simple classification) where explanations aren’t needed. If not specified, effort defaults to medium. | Set reasoning effort: "minimal". Avoid for multi-step planning or tool-heavy workflows. |

**Supported Models:**  
- gpt-5  
- gpt-5-mini  
- gpt-5-nano  

**Supported API Endpoints** 
- Responses API 
- Chat Completions API 

Note: We recommend to use Responses API with GPT-5 series of model to get the most performance out of the models. 

## Prerequisites 

Let's begin with updating your OpenAI SDK that supports the new params and tools for GPT-5. Make sure you've set OPENAI_API_KEY as an environment variable. 

```python
!pip install --quiet --upgrade openai pandas && \
echo -n "openai " && pip show openai | grep '^Version:' | cut -d' ' -f2 && \
echo -n "pandas " && pip show pandas | grep '^Version:' | cut -d' ' -f2
```

    openai 1.99.2
    pandas 2.3.1

## 1. Verbosity Parameter 

### 1.1 Overview 
The verbosity parameter lets you hint the model to be more or less expansive in its replies.   

**Values:** "low", "medium", "high"

- low → terse UX, minimal prose.
- medium (default) → balanced detail.
- high → verbose, great for audits, teaching, or hand-offs.

Keep prompts stable and use the param rather than re-writing.

```python
from openai import OpenAI
import pandas as pd
from IPython.display import display

client = OpenAI()

question = "Write a poem about a boy and his first pet dog."

data = []

for verbosity in ["low", "medium", "high"]:
    response = client.responses.create(
        model="gpt-5-mini",
        input=question,
        text={"verbosity": verbosity}
    )

    # Extract text
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

# Create DataFrame
df = pd.DataFrame(data)

# Display nicely with centered headers
pd.set_option('display.max_colwidth', None)
styled_df = df.style.set_table_styles(
    [
        {'selector': 'th', 'props': [('text-align', 'center')]},  # Center column headers
        {'selector': 'td', 'props': [('text-align', 'left')]}     # Left-align table cells
    ]
)

display(styled_df)
```

The output tokens scale roughly linearly with verbosity: low (560) → medium (849) → high (1288).

### 2.3 Using Verbosity for Coding Use Cases 

The verbosity parameter also influences the length and complexity of generated code, as well as the depth of accompanying explanations. Here's an example, wherein we use various verboisty levels for a task to generate a Python program that sorts an array of 1000000 random numbers. 

```python
from openai import OpenAI

client = OpenAI()

prompt = "Output a Python program that sorts an array of 1000000 random numbers"

def ask_with_verbosity(verbosity: str, question: str):
    response = client.responses.create(
        model="gpt-5-mini",
        input=question,
        text={
            "verbosity": verbosity
        }
    )

    # Extract assistant's text output
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content in item.content:
                if hasattr(content, "text"):
                    output_text += content.text

    # Token usage details
    usage = response.usage

    print("--------------------------------")
    print(f"Verbosity: {verbosity}")
    print("Output:")
    print(output_text)
    print("Tokens => input: {} | output: {}".format(
        usage.input_tokens, usage.output_tokens
    ))


# Example usage:
ask_with_verbosity("low", prompt)
```

    --------------------------------
    Verbosity: low
    Output:
    ```python
    #!/usr/bin/env python3
    import random
    import time
    
    def main():
        N = 1_000_000
        arr = [random.random() for _ in range(N)]
    
        t0 = time.perf_counter()
        arr.sort()
        t1 = time.perf_counter()
    
        print(f"Sorted {N} numbers in {t1 - t0:.4f} seconds")
        print("First 10:", arr[:10])
        print("Last 10:", arr[-10:])
    
    if __name__ == "__main__":
        main()
    ```
    Tokens => input: 21 | output: 575

Notice that the code output is a plain script. Now, lets run with 'medium' 

```python
ask_with_verbosity("medium", prompt)
```

    --------------------------------
    Verbosity: medium
    Output:
    Here's a simple Python script that generates 1,000,000 random numbers, sorts them using the built-in Timsort, and reports timings and a small sample of the sorted output:
    
    ```python
    #!/usr/bin/env python3
    import random
    import time
    
    def main():
        N = 1_000_000
        random.seed(42)  # remove or change for different runs
    
        t0 = time.perf_counter()
        data = [random.random() for _ in range(N)]
        t1 = time.perf_counter()
    
        data.sort()
        t2 = time.perf_counter()
    
        # Basic verification and sample output
        is_sorted = all(data[i] <= data[i+1] for i in range(len(data)-1))
        print(f"Generated {N} random numbers in {t1 - t0:.3f} seconds")
        print(f"Sorted in {t2 - t1:.3f} seconds")
        print("Sorted check:", is_sorted)
        print("First 10 values:", data[:10])
        print("Last 10 values:", data[-10:])
    
    if __name__ == "__main__":
        main()
    ```
    
    Notes:
    - This uses Python's built-in list sort (Timsort), which is efficient for general-purpose sorting.
    - If you need more memory- and performance-efficient numeric operations on large arrays, consider using NumPy (numpy.random.random and numpy.sort).
    Tokens => input: 21 | output: 943

Medium verboisty, generated richer code with additioanl explanations. Let's do the same with high. 

```python
ask_with_verbosity("high", prompt)
```

    --------------------------------
    Verbosity: high
    Output:
    Here's a single, self-contained Python program that generates 1,000,000 random numbers and sorts them. It supports two backends: the built-in Python list sort (Timsort) and NumPy (if you have NumPy installed). It measures and prints the time taken for generation, sorting, and verification.
    
    Copy the code into a file (for example sort_random.py) and run it. By default it uses the pure Python backend; pass --backend numpy to use NumPy.
    
    Note: Sorting a million Python floats uses a moderate amount of memory (Python floats and list overhead). NumPy will typically be faster and use less overhead but requires the numpy package.
    
    Program:
    
    import time
    import random
    import argparse
    import sys
    
    def is_sorted_list(a):
        # Linear check for sortedness
        return all(a[i] <= a[i+1] for i in range(len(a)-1))
    
    def main():
        parser = argparse.ArgumentParser(description="Generate and sort random numbers.")
        parser.add_argument("--n", type=int, default=1_000_000, help="Number of random numbers (default: 1,000,000)")
        parser.add_argument("--backend", choices=["python", "numpy"], default="python",
                            help="Sorting backend to use: 'python' (default) or 'numpy'")
        parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
        parser.add_argument("--sample", type=int, default=10, help="How many sample elements to print (default: 10)")
        args = parser.parse_args()
    
        n = args.n
        backend = args.backend
        seed = args.seed
        sample = args.sample
    
        print(f"Generating {n:,} random numbers using backend: {backend!r}, seed={seed}")
    
        random.seed(seed)
    
        if backend == "python":
            # Generate list of floats in Python
            t0 = time.perf_counter()
            data = [random.random() for _ in range(n)]
            t1 = time.perf_counter()
            gen_time = t1 - t0
            print(f"Generated {n:,} numbers in {gen_time:.4f} s")
    
            if sample > 0:
                print("Sample before sort:", data[:sample])
    
            # Sort in-place
            t0 = time.perf_counter()
            data.sort()
            t1 = time.perf_counter()
            sort_time = t1 - t0
            print(f"Sorted {n:,} numbers in {sort_time:.4f} s (Python list.sort)")
    
            if sample > 0:
                print("Sample after sort: ", data[:sample])
    
            # Verify sortedness
            t0 = time.perf_counter()
            ok = is_sorted_list(data)
            t1 = time.perf_counter()
            verify_time = t1 - t0
            print(f"Verified sortedness in {verify_time:.4f} s -> {'OK' if ok else 'NOT SORTED'}")
    
        else:  # numpy backend
            try:
                import numpy as np
            except ImportError:
                print("NumPy is not installed. Install it with 'pip install numpy' or use the python backend.")
                sys.exit(1)
    
            # Use the new Generator API for reproducible generation
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            data = rng.random(n)  # numpy array of floats
            t1 = time.perf_counter()
            gen_time = t1 - t0
            print(f"Generated {n:,} numbers in {gen_time:.4f} s (NumPy)")
    
            if sample > 0:
                print("Sample before sort:", data[:sample])
    
            # Sort in-place using NumPy's sort
            t0 = time.perf_counter()
            data.sort()  # in-place quicksort/mergesort (NumPy chooses default)
            t1 = time.perf_counter()
            sort_time = t1 - t0
            print(f"Sorted {n:,} numbers in {sort_time:.4f} s (NumPy sort)")
    
            if sample > 0:
                print("Sample after sort: ", data[:sample])
    
            # Verify sortedness
            t0 = time.perf_counter()
            ok = np.all(np.diff(data) >= 0)
            t1 = time.perf_counter()
            verify_time = t1 - t0
            print(f"Verified sortedness in {verify_time:.4f} s -> {'OK' if ok else 'NOT SORTED'}")
    
        print("Done.")
    
    if __name__ == "__main__":
        main()
    
    Usage examples:
    - Pure Python (default):
      python sort_random.py
    
    - NumPy backend (if installed):
      python sort_random.py --backend numpy
    
    - Use a different size:
      python sort_random.py --n 500000
    
    Notes and tips:
    - Pure Python uses random.random in a list comprehension, then list.sort(). Sorting a list of 1,000,000 Python floats is quite feasible but uses more memory than a NumPy array because of Python object overhead.
    - NumPy's random generation and sorting are implemented in C and are typically much faster and more memory efficient for large numeric arrays.
    - You can change the seed to get different random sequences, or omit seed for non-deterministic results.
    - If you plan to sort data that doesn't fit in memory, consider external sorting approaches (merge sort with chunking to disk) or use specialized libraries.
    Tokens => input: 21 | output: 2381

High verbosity yielded additional details and explanations. 

### 1.3 Takeaways 

The new verbosity parameter reliably scales both the length and depth of the model’s output while preserving correctness and reasoning quality - **without changing the underlying prompt**.
In this example:

- **Low verbosity** produces a minimal, functional script with no extra comments or structure.
- **Medium verbosity** adds explanatory comments, function structure, and reproducibility controls.
- **High verbosity** yields a comprehensive, production-ready script with argument parsing, multiple sorting methods, timing/verification, usage notes, and best-practice tips.

## 2. Free‑Form Function Calling

### 2.1 Overview 
GPT‑5 can now send raw text payloads - anything from Python scripts to SQL queries - to your custom tool without wrapping the data in JSON using the new tool `"type": "custom"`. This differs from classic structured function calls, giving you greater flexibility when interacting with external runtimes such as:

- code_exec with sandboxes (Python, C++, Java, …)
- SQL databases
- Shell environments
- Configuration generators

**Note that custom tool type does NOT support parallel tool calling.**

### 2.2 Quick Start Example - Compute the Area of a Circle

The code below produces a simple python code to calculate area of a circle, and instruct the model to use the freeform tool call to output the result. 

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
print(response.output)
```

    [ResponseReasoningItem(id='rs_6894e31b1f8081999d18325e5aeffcfe0861a2e1728d1664', summary=[], type='reasoning', content=[], encrypted_content=None, status=None), ResponseCustomToolCall(call_id='call_Gnqod2MwPvayp2JdNyA0z0Ah', input='# Counting \'r\'s in the word "strawberry" and computing circle area with that radius\nimport math\nr = "strawberry".count(\'r\')\narea = math.pi * r**2\n{"radius": r, "area": area, "area_exact": f"{r}*pi"}', name='code_exec', type='custom_tool_call', id='ctc_6894e31c66f08199abd622bb5ac3c4260861a2e1728d1664', status='completed')]

The model emits a `tool call` containing raw Python. You execute that code server‑side, capture the printed result, and send it back in a follow‑up responses.create call.

### 2.3 Mini‑Benchmark – Sorting an Array in Three Languages
To illustrate the use of free form tool calling, we will ask GPT‑5 to:
- Generate Python, C++, and Java code that sorts a fixed array 10 times.
- Print only the time (in ms) taken for each iteration in the code. 
- Call all three functions, and then stop 

```python
from openai import OpenAI
from typing import List, Optional

MODEL_NAME = "gpt-5"

# Tools that will be passed to every model invocation. They are defined once so
# that the configuration lives in a single place.
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
    """Wrapper around ``client.responses.create``.

    Parameters
    ----------
    input_messages: List[dict]
        The running conversation history to feed to the