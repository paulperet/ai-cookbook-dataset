# Evaluating a New Model on Existing Logged Responses

This guide demonstrates how to evaluate a new model (e.g., `gpt-4.1-mini`) against an older model (e.g., `gpt-4o-mini`) by using your existing response logs. This approach leverages data you've already generated, saving you the effort of building a full evaluation dataset from scratch.

## Prerequisites

Ensure you have the OpenAI Python SDK installed and configured.

```bash
pip install openai
```

You will also need an OpenAI API key set in your environment and an organization where data logging is enabled. To verify logging is active, visit your [logs page](https://platform.openai.com/logs?api=responses).

## Setup

Begin by importing the necessary libraries and initializing the OpenAI client.

```python
import openai
import os

client = openai.OpenAI()
```

## Step 1: Generate Baseline Responses with the Old Model

To perform a comparison, you first need a set of logged responses. We will generate these by asking `gpt-4o-mini` to explain several code files from the OpenAI SDK.

First, locate some example files within the SDK.

```python
openai_sdk_file_path = os.path.dirname(openai.__file__)

file_paths = [
    os.path.join(openai_sdk_file_path, "resources", "evals", "evals.py"),
    os.path.join(openai_sdk_file_path, "resources", "responses", "responses.py"),
    os.path.join(openai_sdk_file_path, "resources", "images.py"),
    os.path.join(openai_sdk_file_path, "resources", "embeddings.py"),
    os.path.join(openai_sdk_file_path, "resources", "files.py"),
]
```

Now, generate an explanation for each file. These calls will create responses that are automatically logged to your account (provided logging is enabled).

```python
for file_path in file_paths:
    response = client.responses.create(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What does this file do?"},
                    {"type": "input_text", "text": open(file_path, "r").read()},
                ],
            }
        ],
        model="gpt-4o-mini",
    )
    print(response.output_text)
```

**Important:** Verify the responses appear on your [logs page](https://platform.openai.com/logs?api=responses). If they do not, data logging may be disabled for your organization.

## Step 2: Define the Evaluation Criteria

You need a robust grader to score the quality of the model explanations. Define a detailed system prompt that instructs an evaluator model (here, `o3`) on how to assess responses.

```python
grader_system_prompt = """
You are **Code-Explanation Grader**, an expert software engineer and technical writer.
Your job is to score how well *Model A* explained the purpose and behaviour of a given source-code file.

### What you receive
1. **File contents** – the full text of the code file (or a representative excerpt).
2. **Candidate explanation** – the answer produced by Model A that tries to describe what the file does.

### What to produce
Return a single JSON object that can be parsed by `json.loads`, containing:
```json
{
  "steps": [
    { "description": "...", "result": "float" },
    { "description": "...", "result": "float" },
    { "description": "...", "result": "float" }
  ],
  "result": "float"
}
```
• Each object in `steps` documents your reasoning for one category listed under “Scoring dimensions”.
• Place your final 1 – 7 quality score (inclusive) in the top-level `result` key as a **string** (e.g. `"5.5"`).

### Scoring dimensions (evaluate in this order)

1. **Correctness & Accuracy ≈ 45 %**
   • Does the explanation match the actual code behaviour, interfaces, edge cases, and side effects?
   • Fact-check every technical claim; penalise hallucinations or missed key functionality.

2. **Completeness & Depth ≈ 25 %**
   • Are all major components, classes, functions, data flows, and external dependencies covered?
   • Depth should be appropriate to the file’s size/complexity; superficial glosses lose points.

3. **Clarity & Organization ≈ 20 %**
   • Is the explanation well-structured, logically ordered, and easy for a competent developer to follow?
   • Good use of headings, bullet lists, and concise language is rewarded.

4. **Insight & Usefulness ≈ 10 %**
   • Does the answer add valuable context (e.g., typical use cases, performance notes, risks) beyond line-by-line paraphrase?
   • Highlighting **why** design choices matter is a plus.

### Error taxonomy
• **Major error** – Any statement that materially misrepresents the file (e.g., wrong API purpose, inventing non-existent behaviour).
• **Minor error** – Small omission or wording that slightly reduces clarity but doesn’t mislead.
List all found errors in your `steps` reasoning.

### Numeric rubric
1  Catastrophically wrong; mostly hallucination or irrelevant.
2  Many major errors, few correct points.
3  Several major errors OR pervasive minor mistakes; unreliable.
4  Mostly correct but with at least one major gap or multiple minors; usable only with caution.
5  Solid, generally correct; minor issues possible but no major flaws.
6  Comprehensive, accurate, and clear; only very small nit-picks.
7  Exceptional: precise, thorough, insightful, and elegantly presented; hard to improve.

Use the full scale. Reserve 6.5 – 7 only when you are almost certain the explanation is outstanding.

Then set `"result": "4.0"` (example).

Be rigorous and unbiased.
"""

user_input_message = """**User input**

{{item.input}}

**Response to evaluate**

{{sample.output_text}}
"""
```

## Step 3: Create the Evaluation

With the grader defined, create an evaluation object that uses your response logs as the data source.

```python
logs_eval = client.evals.create(
    name="Code QA Eval",
    data_source_config={
        "type": "logs",
    },
    testing_criteria=[
        {
            "type": "score_model",
            "name": "General Evaluator",
            "model": "o3",
            "input": [
                {"role": "system", "content": grader_system_prompt},
                {"role": "user", "content": user_input_message},
            ],
            "range": [1, 7],
            "pass_threshold": 5.5,
        }
    ],
)
```

## Step 4: Evaluate the Baseline Model (`gpt-4o-mini`)

Kick off the first evaluation run to score the responses you just generated with `gpt-4o-mini`. This run will fetch the most recent logged responses.

```python
gpt_4o_mini_run = client.evals.runs.create(
    name="gpt-4o-mini",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {"type": "responses", "limit": len(file_paths)},
    },
)
```

## Step 5: Evaluate the New Model (`gpt-4.1-mini`)

Now, evaluate the new model on the *same inputs*. The configuration instructs the system to use the original user inputs from the logs but generate new responses with `gpt-4.1-mini`.

```python
gpt_41_mini_run = client.evals.runs.create(
    name="gpt-4.1-mini",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {"type": "responses", "limit": len(file_paths)},
        "input_messages": {
            "type": "item_reference",
            "item_reference": "item.input",
        },
        "model": "gpt-4.1-mini",
    },
)
```

## Step 6: Review the Results

The evaluation runs will process asynchronously. You can monitor the progress and view the detailed report via the URL provided by the run object.

```python
print(gpt_4o_mini_run.report_url)
```

Visit the provided URL in your browser to see a side-by-side comparison of the scores for `gpt-4o-mini` and `gpt-4.1-mini`. The dashboard will show metrics like average score and pass rate, allowing you to determine if the new model performs better according to your defined grading criteria.

## Summary

You have successfully set up an automated evaluation pipeline using your existing response logs. This method allows for efficient A/B testing of new models against historical data, providing quantitative insights to guide your model selection.