# Structured Output Evaluation Cookbook
 
This notebook walks you through a set of focused, runnable examples how to use the OpenAI **Evals** framework to **test, grade, and iterate on tasks that require large‑language models to produce structured outputs**.

> **Why does this matter?**  
> Production systems often depend on JSON, SQL, or domain‑specific formats.  Relying on spot checks or ad‑hoc prompt tweaks quickly breaks down.  Instead, you can *codify* expectations as automated evals and let your team ship with safety bricks instead of sand.



## Quick Tour

* **Section 1 – Prerequisites**: environment variables and package setup  
* **Section 2 – Walk‑through: Code‑symbol extraction**: end‑to‑end demo that grades the model’s ability to extract function and class names from source code.  We keep the original logic intact and simply layer documentation around it.  
* **Section 3 – Additional Recipes**: sketches of common production patterns such as sentiment extraction as additional code sample for evaluation.
* **Section 4 – Result Exploration**: lightweight helpers for pulling run output and digging into failures.  



## Prerequisites

1. **Install dependencies** (minimum versions shown):

```bash
pip install --upgrade openai
```

2. **Authenticate** by exporting your key:

```bash
export OPENAI_API_KEY="sk‑..."
```

3. **Optional**: if you plan to run evals in bulk, set up an [organization‑level key](https://platform.openai.com/account/org-settings) with appropriate limits.


### Use Case 1: Code symbol extraction


The goal is to **extract all function, class, and constant symbols from python files inside the OpenAI SDK**.  
For each file we ask the model to emit structured JSON like:

```json
{
  "symbols": [
    {"name": "OpenAI", "kind": "class"},
    {"name": "Evals", "kind": "module"},
    ...
  ]
}
```

A rubric model then grades **completeness** (did we capture every symbol?) and **quality** (are the kinds correct?) on a 1‑7 scale.


### Evaluating Code Quality Extraction with a Custom Dataset

Let us walk though an example to evaluate a model's ability to extract symbols from code using the OpenAI **Evals** framework with a custom in-memory dataset.

### Initialize SDK client
Creates an `openai.OpenAI` client using the `OPENAI_API_KEY` we exported above.  Nothing will run without this.


```python
%pip install --upgrade openai pandas rich --quiet



import os
import time
import openai
from rich import print
import pandas as pd

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("_OPENAI_API_KEY"),
)
```

    
    [notice] A new release of pip is available: 24.0 -> 25.1.1
    [notice] To update, run: pip install --upgrade pip
    Note: you may need to restart the kernel to use updated packages.


### Dataset factory & grading rubric
* `get_dataset` builds a small in-memory dataset by reading several SDK files.
* `structured_output_grader` defines a detailed evaluation rubric.
* `client.evals.create(...)` registers the eval with the platform.


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

### Kick off model runs
Here we launch two runs against the same eval: one that calls the **Completions** endpoint, and one that calls the **Responses** endpoint.


```python
### Kick off model runs
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

### Utility poller
Next, we will use a simple loop that waits for all runs to finish, then saves each run’s JSON to disk so you can inspect it later or attach it to CI artifacts.


```python
### Utility poller
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

evalrun_68487dcc749081918ec2571e76cc9ef6 completed
ResultCounts(errored=0, failed=1, passed=0, total=1)

evalrun_68487dcdaba0819182db010fe5331f2e completed
ResultCounts(errored=0, failed=1, passed=0, total=1)


### Load outputs for quick inspection
We will fetch the output items for both runs so we can print or post‑process them.


```python
completions_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_completions_run.id, eval_id=logs_eval.id
)

responses_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_responses_run.id, eval_id=logs_eval.id
)
```

### Human-readable dump
Let us print a side-by-side view of completions vs responses.


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



<h4 style="color: #1CA7EC; font-weight: 600; letter-spacing: 1px; text-shadow: 0 1px 2px rgba(0,0,0,0.08), 0 0px 0px #fff;">
Completions vs Responses Output
</h4>




<style type="text/css">
#T_ac15e th {
  font-size: 1.1em;
  background-color: #323C50;
  color: #FFFFFF;
  border-bottom: 2px solid #1CA7EC;
}
#T_ac15e td {
  font-size: 1em;
  max-width: 650px;
  background-color: #F6F8FA;
  color: #222;
  border-bottom: 1px solid #DDD;
}
#T_ac15e tr:hover td {
  background-color: #D1ECF1;
  color: #18647E;
}
#T_ac15e tbody tr:nth-child(even) td {
  background-color: #E8F1FB;
}
#T_ac15e tbody tr:nth-child(odd) td {
  background-color: #F6F8FA;
}
#T_ac15e table {
  border-collapse: collapse;
  border-radius: 6px;
  overflow: hidden;
}
#T_ac15e_row0_col0, #T_ac15e_row0_col1 {
  white-space: pre-wrap;
  word-break: break-word;
  padding: 8px;
}
</style>
<table id="T_ac15e">
  <thead>
    <tr>
      <th id="T_ac15e_level0_col0" class="col_heading level0 col0" >Completions Output</th>
      <th id="T_ac15e_level0_col1" class="col_heading level0 col1" >Responses Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_ac15e_row0_col0" class="data row0 col0" >{"symbols":[{"name":"Evals","symbol_type":"class"},{"name":"AsyncEvals","symbol_type":"class"},{"name":"EvalsWithRawResponse","symbol_type":"class"},{"name":"AsyncEvalsWithRawResponse","symbol_type":"class"},{"name":"EvalsWithStreamingResponse","symb...</td>
      <td id="T_ac15e_row0_col1" class="data row0 col1" >{"symbols":[{"name":"Evals","symbol_type":"class"},{"name":"runs","symbol_type":"property"},{"name":"with_raw_response","symbol_type":"property"},{"name":"with_streaming_response","symbol_type":"property"},{"name":"create","symbol_type":"function"},{...</td>
    </tr>
  </tbody>
</table>



### Use Case 2: Multi-lingual Sentiment Extraction
In a similar way, let us evaluate a multi-lingual sentiment extraction model with structured outputs.


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
        "text": "总体来说，我对这款产品很