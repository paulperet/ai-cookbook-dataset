# Tool Evaluation with OpenAI Evals: Extracting Symbols from Python Code

This guide demonstrates how to use OpenAI Evals to measure and improve a model's ability to extract structured information—specifically, symbols like functions, classes, and variables—from Python source code.

## Prerequisites

Before you begin, ensure you have the following:

1.  An OpenAI API key with access to the required models.
2.  Python 3.7 or later installed on your system.

## Step 1: Install Dependencies and Set Up the Client

First, install the necessary Python packages and configure the OpenAI client.

```bash
pip install --upgrade openai pandas jinja2 rich
```

```python
import os
import time
import json
import openai
import pandas as pd
from IPython.display import display, HTML

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("_OPENAI_API_KEY"),
)
```

## Step 2: Create a Dataset from Local Files

We'll build a small evaluation dataset by reading Python files from the OpenAI SDK itself. This function reads a few specific files and returns their content.

```python
def get_dataset(limit=None):
    # Locate the openai SDK directory
    openai_sdk_file_path = os.path.dirname(openai.__file__)

    # Define the specific files to read
    file_paths = [
        os.path.join(openai_sdk_file_path, "resources", "evals", "evals.py"),
        os.path.join(openai_sdk_file_path, "resources", "responses", "responses.py"),
        os.path.join(openai_sdk_file_path, "resources", "images.py"),
        os.path.join(openai_sdk_file_path, "resources", "embeddings.py"),
        os.path.join(openai_sdk_file_path, "resources", "files.py"),
    ]

    # Read each file and store its content
    items = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            items.append({"input": f.read()})
    if limit:
        return items[:limit]
    return items
```

## Step 3: Define the Evaluation Rubric

An evaluation needs clear grading criteria. We define a system prompt that instructs the evaluator model on how to score the quality of extracted symbols.

```python
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
{{sample.output_tools[0].function.arguments.symbols}}
</Extracted Information>
"""
```

## Step 4: Create the Evaluation

Now, we create the evaluation object on the OpenAI platform. This defines the task, the data schema, and the scoring model.

```python
logs_eval = client.evals.create(
    name="Code QA Eval",
    data_source_config={
        "type": "custom",
        "item_schema": {"type": "object", "properties": {"input": {"type": "string"}}},
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
            "pass_threshold": 5.0,
        }
    ],
)
```

## Step 5: Define the Tool for Symbol Extraction

The model will use a function-calling tool to structure its output. Here we define the schema for the `extract_symbols` tool.

```python
symbol_tool = {
    "name": "extract_symbols",
    "description": "Extract the symbols from the code file",
    "parameters": {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "description": "A list of symbols extracted from Python code.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the symbol."},
                        "symbol_type": {"type": "string", "description": "The type of the symbol, e.g., variable, function, class."},
                    },
                    "required": ["name", "symbol_type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["symbols"],
        "additionalProperties": False,
    },
}
```

## Step 6: Launch Model Evaluation Runs

We will run the same evaluation task using two different OpenAI endpoints and models to compare their performance.

```python
# Run 1: Using the Completions endpoint with GPT-4.1
gpt_4one_completions_run = client.evals.runs.create(
    name="gpt-4.1",
    eval_id=logs_eval.id,
    data_source={
        "type": "completions",
        "source": {"type": "file_content", "content": [{"item": item} for item in get_dataset(limit=1)]},
        "input_messages": {
            "type": "template",
            "template": [
                {"type": "message", "role": "system", "content": {"type": "input_text", "text": "You are a helpful assistant."}},
                {"type": "message", "role": "user", "content": {"type": "input_text", "text": "Extract the symbols from the code file {{item.input}}"}},
            ],
        },
        "model": "gpt-4.1",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "tools": [{"type": "function", "function": symbol_tool}],
        },
    },
)

# Run 2: Using the Responses endpoint with GPT-4.1-mini
gpt_4one_responses_run = client.evals.runs.create(
    name="gpt-4.1-mini",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {"type": "file_content", "content": [{"item": item} for item in get_dataset(limit=1)]},
        "input_messages": {
            "type": "template",
            "template": [
                {"type": "message", "role": "system", "content": {"type": "input_text", "text": "You are a helpful assistant."}},
                {"type": "message", "role": "user", "content": {"type": "input_text", "text": "Extract the symbols from the code file {{item.input}}"}},
            ],
        },
        "model": "gpt-4.1-mini",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "tools": [{"type": "function", **symbol_tool}],
        },
    },
)
```

## Step 7: Poll for Run Completion

Evaluation runs are asynchronous. This helper function polls their status until they are complete.

```python
def poll_runs(eval_id, run_ids):
    # Poll both runs at the same time, until they are complete or failed
    while True:
        runs = [client.evals.runs.retrieve(run_id, eval_id=eval_id) for run_id in run_ids]
        for run in runs:
            print(run.id, run.status, run.result_counts)
        if all(run.status in ("completed", "failed") for run in runs):
            break
        time.sleep(5)

# Wait for our two runs to finish
poll_runs(logs_eval.id, [gpt_4one_completions_run.id, gpt_4one_responses_run.id])
```

## Step 8: Retrieve and Inspect the Results

Once the runs are complete, we can fetch the outputs and examine the symbols extracted by each model.

```python
# Fetch the output items for each run
completions_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_completions_run.id, eval_id=logs_eval.id
)

responses_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_responses_run.id, eval_id=logs_eval.id
)

# Helper function to parse the symbols from the tool call output
def extract_symbols(output_list):
    symbols_list = []
    for item in output_list:
        try:
            args = item.sample.output[0].tool_calls[0]["function"]["arguments"]
            symbols = json.loads(args)["symbols"]
            symbols_list.append(symbols)
        except Exception as e:
            symbols_list.append([{"error": str(e)}])
    return symbols_list

# Extract symbols from both runs
completions_symbols = extract_symbols(completions_output)
responses_symbols = extract_symbols(responses_output)
```

## Step 9: Compare the Extracted Symbols

Let's create a simple visual comparison of the symbols extracted by the Completions endpoint (GPT-4.1) versus the Responses endpoint (GPT-4.1-mini).

```python
def symbols_to_html_table(symbols):
    if symbols and isinstance(symbols, list):
        df = pd.DataFrame(symbols)
        return (
            df.style
            .set_properties(**{
                'white-space': 'pre-wrap',
                'word-break': 'break-word',
                'padding': '2px 6px',
                'border': '1px solid #C3E7FA',
                'font-size': '0.92em',
                'background-color': '#FDFEFF'
            })
            .set_table_styles([{
                'selector': 'th',
                'props': [
                    ('font-size', '0.95em'),
                    ('background-color', '#1CA7EC'),
                    ('color', '#fff'),
                    ('border-bottom', '1px solid #18647E'),
                    ('padding', '2px 6px')
                ]
            }])
            .hide(axis='index')
            .to_html()
        )
    return f"<div style='padding:4px 0;color:#D9534F;font-style:italic;font-size:0.9em'>{str(symbols)}</div>"

# Build an HTML table for side-by-side comparison
table_rows = []
max_len = max(len(completions_symbols), len(responses_symbols))
for i in range(max_len):
    c_html = symbols_to_html_table(completions_symbols[i]) if i < len(completions_symbols) else ""
    r_html = symbols_to_html_table(responses_symbols[i]) if i < len(responses_symbols) else ""
    table_rows.append(f"""
      <tr style="height:1.2em;">
          <td style="vertical-align:top; background:#F6F8FA; border-right:1px solid #E3E3E3; padding:2px 4px;">{c_html}</td>
          <td style="vertical-align:top; background:#F6F8FA; padding:2px 4px;">{r_html}</td>
      </tr>
    """)

table_html = f"""
<div style="margin-bottom:0.5em;margin-top:0.2em;">
  <h4 style="color:#1CA7EC;font-weight:600;letter-spacing:0.5px;
     text-shadow:0 1px 2px rgba(0,0,0,0.06), 0 0px 0px #fff;font-size:1.05em;margin:0 0 0.35em 0;">
    Completions vs Responses Output Symbols
  </h4>
  <table style="border-collapse:separate;border-spacing:0 0.2em;width:100%;border-radius:5px;overflow:hidden;box-shadow:0 1px 7px #BEE7FA22;">
    <thead>
      <tr style="height:1.4em;">
              <th style="width:50%;background:#323C50;color:#fff;font-size:1em;padding:6px 10px;border-bottom:2px solid #1CA7EC;text-align:center;">Completions Output</th>
      <th style="width:50%;background:#323C50;color:#fff;font-size:1em;padding:6px 10px;border-bottom:2px solid #1CA7EC;text-align:center;">Responses Output</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</div>
"""

display(HTML(table_html))
```

## Step 10: Review Results in the Evals Dashboard

For a comprehensive analysis, navigate to the **Evals Dashboard** in the OpenAI platform. The dashboard provides:
*   A visual overview of all runs and their scores.
*   Detailed explanations for any failed evaluations.
*   Aggregated metrics to help you understand model performance.

## Conclusion

This tutorial demonstrated how to use OpenAI Evals to create a reproducible framework for evaluating an LLM's ability to perform structured information extraction from code. By defining a clear tool schema, a rigorous grading rubric, and a custom dataset, you can quantitatively measure and iteratively improve model performance on this task.

**Next Steps:**
*   Expand your dataset with more diverse code files.
*   Adjust the grading rubric or pass threshold based on your quality requirements.
*   Use the evaluation results to fine-tune your prompts or model selection.

For more details, see the [OpenAI Evals documentation](https://platform.openai.com/docs/guides/evals).