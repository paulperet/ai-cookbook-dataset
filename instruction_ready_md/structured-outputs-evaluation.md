# Structured Output Evaluation Cookbook

This guide walks you through a set of focused, runnable examples on how to use the OpenAI **Evals** framework to **test, grade, and iterate on tasks that require large-language models to produce structured outputs**.

> **Why does this matter?**  
> Production systems often depend on JSON, SQL, or domain-specific formats. Relying on spot checks or ad-hoc prompt tweaks quickly breaks down. Instead, you can *codify* expectations as automated evals and let your team ship with safety bricks instead of sand.

## Quick Tour

* **Section 1 – Prerequisites**: Environment variables and package setup.
* **Section 2 – Walk-through: Code-symbol extraction**: An end-to-end demo that grades the model’s ability to extract function and class names from source code.
* **Section 3 – Additional Recipes**: Sketches of common production patterns, such as sentiment extraction.
* **Section 4 – Result Exploration**: Lightweight helpers for pulling run output and digging into failures.

---

## 1. Prerequisites

### 1.1 Install Dependencies
Install the required packages using pip.

```bash
pip install --upgrade openai pandas rich
```

### 1.2 Set Your API Key
Authenticate by exporting your OpenAI API key to your environment.

```bash
export OPENAI_API_KEY="sk-..."
```

### 1.3 Optional: Organization-Level Key
If you plan to run evals in bulk, consider setting up an [organization-level key](https://platform.openai.com/account/org-settings) with appropriate limits.

---

## 2. Walk-through: Code Symbol Extraction

The goal is to **extract all function, class, and constant symbols from Python files inside the OpenAI SDK**. For each file, we ask the model to emit structured JSON like:

```json
{
  "symbols": [
    {"name": "OpenAI", "kind": "class"},
    {"name": "Evals", "kind": "module"}
  ]
}
```

A rubric model then grades **completeness** (did we capture every symbol?) and **quality** (are the kinds correct?) on a 1–7 scale.

### 2.1 Initialize the SDK Client
First, import the necessary libraries and create an `openai.OpenAI` client. This client will be used for all subsequent API calls.

```python
import os
import time
import openai
from rich import print
import pandas as pd

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("_OPENAI_API_KEY"),
)
```

### 2.2 Create a Custom Dataset and Grading Rubric
Next, you'll define a function to build a small, in-memory dataset by reading several files from the OpenAI SDK. You'll also define a detailed evaluation rubric that the grader model will use.

```python
def get_dataset(limit=None):
    openai_sdk_file_path = os.path.dirname(openai.__file__)

    file_paths = [
        os.path.join(openai_sdk_file_path, "resources", "evals", "evals.py"),
        os.path.join(openai_sdk_file_path, "resources", "responses", "responses.py"),
        os.path.join(openai_sdk_file_path, "resources", "images.py"),
        os.path.join(openai_sdk_file_path, "resources", "embeddings.py"),
        os.path.join(openai_sdk_file_path, "resources", "files.py"),
    ]

    items = []
    for file_path in file_paths:
        items.append({"input": open(file_path, "r").read()})
    if limit:
        return items[:limit]
    return items


structured_output_grader = """
You are a helpful assistant that grades the quality of extracted information from a code file.
You will be given a code file and a list of extracted information.
You should grade the quality of the extracted information.

You should grade the quality on a scale of 1 to 7.
You should apply the following criteria, and calculate your score as follows:
You should first check for completeness on a scale of 1 to 7.
Then you should apply a quality modifier.

The quality modifier is a multiplier from 0 to 1 that you multiply by the completeness score.
If there is 100% coverage for completion and it is all high quality, then you would return 7*1.
If there is 100% coverage for completion but it is all low quality, then you would return 7*0.5.
etc.
"""

structured_output_grader_user_prompt = """
<Code File>
{{item.input}}
</Code File>

<Extracted Information>
{{sample.output_json.symbols}}
</Extracted Information>
"""
```

### 2.3 Register the Evaluation
Now, register the evaluation with the platform using `client.evals.create`. This defines the data source and the testing criteria (the grader model).

```python
logs_eval = client.evals.create(
    name="Code QA Eval",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
        },
        "include_sample_schema": True,
    },
    testing_criteria=[
        {
            "type": "score_model",
            "name": "General Evaluator",
            "model": "o3",
            "input": [
                {"role": "system", "content": structured_output_grader},
                {"role": "user", "content": structured_output_grader_user_prompt},
            ],
            "range": [1, 7],
            "pass_threshold": 5.5,
        }
    ],
)
```

### 2.4 Launch Model Runs
With the eval registered, you can launch runs. Here, you'll create two runs against the same eval: one using the **Completions** endpoint and one using the **Responses** endpoint. Each run will process a limited dataset.

```python
gpt_4one_completions_run = client.evals.runs.create(
    name="gpt-4.1",
    eval_id=logs_eval.id,
    data_source={
        "type": "completions",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in get_dataset(limit=1)],
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "system",
                    "content": {"type": "input_text", "text": "You are a helpful assistant."},
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text": "Extract the symbols from the code file {{item.input}}",
                    },
                },
            ],
        },
        "model": "gpt-4.1",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "python_symbols",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "description": "A list of symbols extracted from Python code.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "The name of the symbol."},
                                        "symbol_type": {
                                            "type": "string", "description": "The type of the symbol, e.g., variable, function, class.",
                                        },
                                    },
                                    "required": ["name", "symbol_type"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["symbols"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        },
    },
)

gpt_4one_responses_run = client.evals.runs.create(
    name="gpt-4.1-mini",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in get_dataset(limit=1)],
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "system",
                    "content": {"type": "input_text", "text": "You are a helpful assistant."},
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text": "Extract the symbols from the code file {{item.input}}",
                    },
                },
            ],
        },
        "model": "gpt-4.1-mini",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "python_symbols",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "description": "A list of symbols extracted from Python code.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "The name of the symbol."},
                                        "symbol_type": {
                                            "type": "string",
                                            "description": "The type of the symbol, e.g., variable, function, class.",
                                        },
                                    },
                                    "required": ["name", "symbol_type"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["symbols"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        },
    },
)
```

### 2.5 Poll for Run Completion
Use a simple polling function to wait for all runs to finish. Once completed, the results are saved to JSON files for later inspection.

```python
def poll_runs(eval_id, run_ids):
    while True:
        runs = [client.evals.runs.retrieve(rid, eval_id=eval_id) for rid in run_ids]
        for run in runs:
            print(run.id, run.status, run.result_counts)
        if all(run.status in {"completed", "failed"} for run in runs):
            # dump results to file
            for run in runs:
                with open(f"{run.id}.json", "w") as f:
                    f.write(
                        client.evals.runs.output_items.list(
                            run_id=run.id, eval_id=eval_id
                        ).model_dump_json(indent=4)
                    )
            break
        time.sleep(5)

poll_runs(logs_eval.id, [gpt_4one_completions_run.id, gpt_4one_responses_run.id])
```

**Example Output:**
```
evalrun_68487dcc749081918ec2571e76cc9ef6 completed ResultCounts(errored=0, failed=1, passed=0, total=1)
evalrun_68487dcdaba0819182db010fe5331f2e completed ResultCounts(errored=0, failed=1, passed=0, total=1)
```

### 2.6 Load and Inspect Outputs
Fetch the output items for both runs so you can examine the results.

```python
completions_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_completions_run.id, eval_id=logs_eval.id
)

responses_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_responses_run.id, eval_id=logs_eval.id
)
```

### 2.7 Display Results Side-by-Side
Create a human-readable, side-by-side view of the outputs from the Completions and Responses runs.

```python
from IPython.display import display, HTML

# Collect outputs for both runs
completions_outputs = [item.sample.output[0].content for item in completions_output]
responses_outputs = [item.sample.output[0].content for item in responses_output]

# Create DataFrame for side-by-side display (truncated to 250 chars for readability)
df = pd.DataFrame({
    "Completions Output": [c[:250].replace('\n', ' ') + ('...' if len(c) > 250 else '') for c in completions_outputs],
    "Responses Output": [r[:250].replace('\n', ' ') + ('...' if len(r) > 250 else '') for r in responses_outputs]
})

# Custom color scheme
custom_styles = [
    {'selector': 'th', 'props': [('font-size', '1.1em'), ('background-color', '#323C50'), ('color', '#FFFFFF'), ('border-bottom', '2px solid #1CA7EC')]},
    {'selector': 'td', 'props': [('font-size', '1em'), ('max-width', '650px'), ('background-color', '#F6F8FA'), ('color', '#222'), ('border-bottom', '1px solid #DDD')]},
    {'selector': 'tr:hover td', 'props': [('background-color', '#D1ECF1'), ('color', '#18647E')]},
    {'selector': 'tbody tr:nth-child(even) td', 'props': [('background-color', '#E8F1FB')]},
    {'selector': 'tbody tr:nth-child(odd) td', 'props': [('background-color', '#F6F8FA')]},
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border-radius', '6px'), ('overflow', 'hidden')]},
]

styled = (
    df.style
    .set_properties(**{'white-space': 'pre-wrap', 'word-break': 'break-word', 'padding': '8px'})
    .set_table_styles(custom_styles)
    .hide(axis="index")
)

display(HTML("""
<h4 style="color: #1CA7EC; font-weight: 600; letter-spacing: 1px; text-shadow: 0 1px 2px rgba(0,0,0,0.08), 0 0px 0px #fff;">
Completions vs Responses Output
</h4>
"""))
display(styled)
```

---

## 3. Additional Recipes

### 3.1 Multi-lingual Sentiment Extraction
You can apply the same evaluation pattern to other structured output tasks. Here is a sketch for evaluating a multi-lingual sentiment extraction model.

```python
# Sample in-memory dataset for sentiment extraction
sentiment_dataset = [
    {
        "text": "I love this product!",
        "channel": "twitter",
        "language": "en"
    },
    {
        "text": "This is the worst experience I've ever had.",
        "channel": "support_ticket",
        "language": "en"
    },
    {
        "text": "It's okay – not great but not bad either.",
        "channel": "app_review",
        "language": "en"
    },
    {
        "text": "No estoy seguro de lo que pienso sobre este producto.",
        "channel": "facebook",
        "language": "es"
    },
    {
        "text": "总体来说，我对这款产品很满意",
        "channel": "weibo",
        "language": "zh"
    }
]
```

To build a full eval, you would:
1. Define a JSON schema for the expected output (e.g., `{"sentiment": "positive", "confidence": 0.95}`).
2. Create a custom dataset from the list above.
3. Register an eval with a grader that checks sentiment accuracy and confidence calibration.
4. Launch runs with models like `gpt-4` or `gpt-3.5-turbo` and compare their performance.

---

## 4. Result Exploration

The polling function in Section 2.5 saves each run's results to a JSON file. You can load these files for deeper analysis.

```python
import json

# Load a saved run result
with open("evalrun_68487dcc749081918ec2571e76cc9ef6.json", "r") as f:
    run_data = json.load(f)

# Inspect the structure
print(json.dumps(run_data, indent=2)[:1000])  # Print first 1000 characters
```

Key fields to explore:
* `sample.input`: The original input (e.g., code file content).
* `sample.output`: The model's generated output.
* `sample.scores`: The scores assigned by the grader model.
* `result_counts`: Summary statistics for the entire run.

Use this data to identify common failure modes, calculate aggregate metrics, or feed results into a CI/CD pipeline.