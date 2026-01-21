# Evaluating MCP-Based Answers with a Custom Dataset

This guide demonstrates how to evaluate a model's ability to answer questions about the [tiktoken](https://github.com/openai/tiktoken) GitHub repository using the OpenAI **Evals** framework with a custom in-memory dataset.

We will compare two models, `gpt-4.1` and `o4-mini`, that leverage the **Model Context Protocol (MCP)** tool for repository-aware, contextually accurate answers.

**Goals:**
- Set up and run an evaluation using OpenAI Evals with a custom dataset.
- Compare the performance of different models leveraging MCP-based tools.
- Provide best practices for professional, reproducible evaluation workflows.

## Prerequisites

Ensure you have the latest OpenAI Python client installed.

```bash
pip install --upgrade openai
```

## 1. Environment Setup

Begin by importing the required libraries and configuring the OpenAI client.

```python
import os
import time
from openai import OpenAI

# Instantiate the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("_OPENAI_API_KEY"),
)
```

## 2. Define the Custom Evaluation Dataset

We define a small, in-memory dataset of question-answer pairs about the `tiktoken` repository. This dataset will test the models' ability to provide accurate and relevant answers with the help of the MCP tool.

Each item contains a `query` (the user’s question) and an `answer` (the expected ground truth). You can modify or extend this dataset to suit your own use case.

```python
def get_dataset(limit=None):
    items = [
        {
            "query": "What is tiktoken?",
            "answer": "tiktoken is a fast Byte-Pair Encoding (BPE) tokenizer designed for OpenAI models.",
        },
        {
            "query": "How do I install the open-source version of tiktoken?",
            "answer": "Install it from PyPI with `pip install tiktoken`.",
        },
        {
            "query": "How do I get the tokenizer for a specific OpenAI model?",
            "answer": 'Call tiktoken.encoding_for_model("<model-name>"), e.g. tiktoken.encoding_for_model("gpt-4o").',
        },
        {
            "query": "How does tiktoken perform compared to other tokenizers?",
            "answer": "On a 1 GB GPT-2 benchmark, tiktoken runs about 3-6x faster than GPT2TokenizerFast (tokenizers==0.13.2, transformers==4.24.0).",
        },
        {
            "query": "Why is Byte-Pair Encoding (BPE) useful for language models?",
            "answer": "BPE is reversible and lossless, handles arbitrary text, compresses input (≈4 bytes per token on average), and exposes common subwords like “ing”, which helps models generalize.",
        },
    ]
    return items[:limit] if limit else items
```

## 3. Define Grading Logic

To evaluate the model’s answers, we use two graders:

- **Pass/Fail Grader (LLM-based):** An LLM-based grader that checks if the model’s answer matches the expected answer (ground truth) or conveys the same meaning.
- **Python MCP Grader:** A Python function that checks whether the model actually used the MCP tool during its response (for auditing tool usage).

> **Best Practice:** Using both LLM-based and programmatic graders provides a more robust and transparent evaluation.

```python
# LLM-based pass/fail grader
pass_fail_grader = """
You are a helpful assistant that grades the quality of the answer to a query about a GitHub repo.
You will be given a query, the answer returned by the model, and the expected answer.
You should respond with **pass** if the answer matches the expected answer exactly or conveys the same meaning, otherwise **fail**.
"""

# User prompt template for the grader
pass_fail_grader_user_prompt = """
<Query>
{{item.query}}
</Query>

<Web Search Result>
{{sample.output_text}}
</Web Search Result>

<Ground Truth>
{{item.answer}}
</Ground Truth>
"""

# Python grader: checks if the MCP tool was used
python_mcp_grader = {
    "type": "python",
    "name": "Assert MCP was used",
    "image_tag": "2025-05-08",
    "pass_threshold": 1.0,
    "source": """
def grade(sample: dict, item: dict) -> float:
    output = sample.get('output_tools', [])
    return 1.0 if len(output) > 0 else 0.0
""",
}
```

## 4. Define the Evaluation Configuration

Now, configure the evaluation using the OpenAI Evals framework. This step specifies the evaluation name, dataset schema, graders, and passing criteria.

> **Best Practice:** Clearly defining your evaluation schema and grading logic up front ensures reproducibility and transparency.

```python
# Create the evaluation definition
logs_eval = client.evals.create(
    name="MCP Eval",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "answer": {"type": "string"},
            },
        },
        "include_sample_schema": True,
    },
    testing_criteria=[
        {
            "type": "label_model",
            "name": "General Evaluator",
            "model": "o3",
            "input": [
                {"role": "system", "content": pass_fail_grader},
                {"role": "user", "content": pass_fail_grader_user_prompt},
            ],
            "passing_labels": ["pass"],
            "labels": ["pass", "fail"],
        },
        python_mcp_grader
    ],
)
```

## 5. Run Evaluations for Each Model

Run the evaluation for each model (`gpt-4.1` and `o4-mini`). Each run is configured to use the MCP tool for repository-aware answers, the same dataset, and consistent evaluation parameters for a fair comparison.

> **Best Practice:** Keeping the evaluation setup consistent across models ensures results are comparable and reliable.

### 5.1 Run Evaluation for `gpt-4.1`

```python
# Run 1: gpt-4.1 using MCP
gpt_4one_responses_run = client.evals.runs.create(
    name="gpt-4.1",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in get_dataset()],
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "system",
                    "content": {
                        "type": "input_text",
                        "text": "You are a helpful assistant that searches the web and gives contextually relevant answers. Never use your tools to answer the query.",
                    },
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text": "Search the web for the answer to the query {{item.query}}",
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
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "gitmcp",
                    "server_url": "https://gitmcp.io/openai/tiktoken",
                    "allowed_tools": [
                        "search_tiktoken_documentation",
                        "fetch_tiktoken_documentation",
                    ],
                    "require_approval": "never",
                }
            ],
        },
    },
)
```

### 5.2 Run Evaluation for `o4-mini`

```python
# Run 2: o4-mini using MCP
gpt_o4_mini_responses_run = client.evals.runs.create(
    name="o4-mini",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in get_dataset()],
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "system",
                    "content": {
                        "type": "input_text",
                        "text": "You are a helpful assistant that searches the web and gives contextually relevant answers.",
                    },
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text": "Search the web for the answer to the query {{item.query}}",
                    },
                },
            ],
        },
        "model": "o4-mini",
        "sampling_params": {
            "seed": 42,
            "max_completions_tokens": 10000,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "gitmcp",
                    "server_url": "https://gitmcp.io/openai/tiktoken",
                    "allowed_tools": [
                        "search_tiktoken_documentation",
                        "fetch_tiktoken_documentation",
                    ],
                    "require_approval": "never",
                }
            ],
        },
    },
)
```

## 6. Poll for Completion and Retrieve Outputs

After launching the evaluation runs, poll until they are complete. This ensures we analyze results only after all model responses have been processed.

> **Best Practice:** Polling with a delay avoids excessive API calls and ensures efficient resource usage.

```python
def poll_runs(eval_id, run_ids):
    while True:
        runs = [client.evals.runs.retrieve(rid, eval_id=eval_id) for rid in run_ids]
        for run in runs:
            print(run.id, run.status, run.result_counts)
        if all(run.status in {"completed", "failed"} for run in runs):
            break
        time.sleep(5)
    
# Start polling both runs.
poll_runs(logs_eval.id, [gpt_4one_responses_run.id, gpt_o4_mini_responses_run.id])
```

The polling output will indicate when runs are complete, for example:

```
evalrun_684769b577488191863b5a51cf4db57a completed ResultCounts(errored=0, failed=5, passed=0, total=5)
evalrun_684769c1ad9c8191affea5aa02ef1215 completed ResultCounts(errored=0, failed=3, passed=2, total=5)
```

## 7. Display and Interpret Model Outputs

Finally, display the outputs from each model for manual inspection and further analysis. You can compare the outputs side-by-side to assess quality, relevance, and correctness.

```python
four_one_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_responses_run.id, eval_id=logs_eval.id
)

o4_mini_output = client.evals.runs.output_items.list(
    run_id=gpt_o4_mini_responses_run.id, eval_id=logs_eval.id
)

print('# gpt‑4.1 Output')
for item in four_one_output:
    print(item.sample.output[0].content)

print('\n# o4-mini Output')
for item in o4_mini_output:
    print(item.sample.output[0].content)
```

### Interpreting the Results

The evaluation results show a clear difference in performance and tool usage:

- The `gpt-4.1` model was explicitly instructed to never use its tools to answer the query, so it never called the MCP server. Consequently, it failed all five evaluation items.
- The `o4-mini` model was not forbidden from using tools and called the MCP server three times. It passed two out of five items.
- Notably, the one example that the `o4-mini` model failed was one where the MCP tool was not used, highlighting the importance of tool utilization for accurate, context-aware answers.

This demonstrates that proper tool usage is critical for models to provide accurate, repository-specific information. For a comprehensive breakdown of the evaluation metrics and results, you can navigate to the "Data" tab in the OpenAI Evals Dashboard.