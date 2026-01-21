# Guide: Evaluating Web Search Quality with a Custom Dataset

This guide demonstrates how to evaluate a model's ability to retrieve correct answers from the web using the OpenAI **Evals** framework with a custom in-memory dataset.

**Goals:**
- Set up and run an evaluation for web search quality.
- Provide a template for evaluating the information retrieval capabilities of LLMs.

## Prerequisites

Ensure you have the necessary libraries installed and your OpenAI API key configured.

### 1. Install and Upgrade Required Packages

```bash
pip install --upgrade openai pandas
```

### 2. Import Libraries and Configure the Client

```python
import os
import time
import pandas as pd
from IPython.display import display
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("_OPENAI_API_KEY"),
)
```

## Step 1: Define Your Custom Evaluation Dataset

You will create a small, in-memory dataset of question-answer pairs. This dataset serves as the ground truth for evaluating the model's web search responses.

Define a function that returns a list of dictionaries, each containing a `query` (the user's search prompt) and an `answer` (the expected correct response).

```python
def get_dataset(limit=None):
    dataset = [
        {
            "query": "coolest person in the world, the 100m dash at the 2008 olympics was the best sports event of all time",
            "answer": "usain bolt",
        },
        {
            "query": "best library in the world, there is nothing better than a dataframe",
            "answer": "pandas",
        },
        {
            "query": "most fun place to visit, I am obsessed with the Philbrook Museum of Art",
            "answer": "tulsa, oklahoma",
        },
        {
            "query": "who created the python programming language, beloved by data scientists everywhere",
            "answer": "guido van rossum",
        },
        {
            "query": "greatest chess player in history, famous for the 1972 world championship",
            "answer": "bobby fischer",
        },
        {
            "query": "the city of lights, home to the eiffel tower and louvre museum",
            "answer": "paris",
        },
        {
            "query": "most popular search engine, whose name is now a verb",
            "answer": "google",
        },
        {
            "query": "the first man to walk on the moon, giant leap for mankind",
            "answer": "neil armstrong",
        },
        {
            "query": "groundbreaking electric car company founded by elon musk",
            "answer": "tesla",
        },
        {
            "query": "founder of microsoft, philanthropist and software pioneer",
            "answer": "bill gates",
        },
    ]
    return dataset[:limit] if limit else dataset
```

**Tip:** You can modify or extend this dataset to test broader search scenarios relevant to your use case.

## Step 2: Define the Grading Logic

To evaluate the model's answers, you will use an LLM-based grader. This grader checks if the model's answer (from the web search) matches or contains the expected ground truth.

Define the system prompt and user prompt template for the grader.

```python
# System prompt for the grader
pass_fail_grader = """
You are a helpful assistant that grades the quality of a web search.
You will be given a query and an answer.
You should grade the quality of the web search.

You should either say "pass" or "fail", if the query contains the answer.
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
```

**Best Practice:** Using an LLM-based grader provides flexibility for evaluating open-ended or nuanced responses where exact string matching is insufficient.

## Step 3: Configure the Evaluation

Now, configure the evaluation using the OpenAI Evals framework. This involves creating an evaluation definition that specifies the dataset schema, the grader, and the passing criteria.

```python
# Create the evaluation definition
logs_eval = client.evals.create(
    name="Web-Search Eval",
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
            "name": "Web Search Evaluator",
            "model": "o3",
            "input": [
                {
                    "role": "system",
                    "content": pass_fail_grader,
                },
                {
                    "role": "user",
                    "content": pass_fail_grader_user_prompt,
                },
            ],
            "passing_labels": ["pass"],
            "labels": ["pass", "fail"],
        }
    ],
)
```

This configuration sets up an evaluation named "Web-Search Eval" that uses your custom dataset and the LLM-based grader you defined.

## Step 4: Run the Evaluation for Your Models

You will now run the evaluation for two models: `gpt-4.1` and `gpt-4.1-mini`. Each run instructs the model to perform a web search for each query in your dataset.

### 4.1 Run Evaluation for GPT-4.1

```python
# Launch the evaluation run for gpt-4.1 using web search
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
        "model": "gpt-4.1",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "tools": [{"type": "web_search_preview"}],
        },
    },
)
```

### 4.2 Run Evaluation for GPT-4.1-mini

```python
# Launch the evaluation run for gpt-4.1-mini using web search
gpt_4one_mini_responses_run = client.evals.runs.create(
    name="gpt-4.1-mini",
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
        "model": "gpt-4.1-mini",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "tools": [{"type": "web_search_preview"}],
        },
    },
)
```

## Step 5: Poll for Run Completion

Evaluation runs are asynchronous. You need to poll their status until they are complete.

Define a helper function to poll multiple runs and then call it for your two runs.

```python
def poll_runs(eval_id, run_ids):
    """Poll evaluation runs until they are completed or failed."""
    while True:
        runs = [client.evals.runs.retrieve(run_id, eval_id=eval_id) for run_id in run_ids]
        for run in runs:
            print(run.id, run.status, run.result_counts)
        if all(run.status in {"completed", "failed"} for run in runs):
            break
        time.sleep(5)

# Start polling the runs until completion
poll_runs(logs_eval.id, [gpt_4one_responses_run.id, gpt_4one_mini_responses_run.id])
```

Once complete, you will see output similar to:
```
evalrun_... completed ResultCounts(errored=0, failed=1, passed=9, total=10)
evalrun_... completed ResultCounts(errored=0, failed=0, passed=10, total=10)
```

## Step 6: Retrieve and Display Model Outputs

Finally, retrieve the outputs from both model runs and display them side-by-side for manual inspection and analysis.

```python
# Retrieve output items for the GPT-4.1 model
four_one = client.evals.runs.output_items.list(
    run_id=gpt_4one_responses_run.id, eval_id=logs_eval.id
)

# Retrieve output items for the GPT-4.1-mini model
four_one_mini = client.evals.runs.output_items.list(
    run_id=gpt_4one_mini_responses_run.id, eval_id=logs_eval.id
)

# Extract the text content from the outputs
four_one_outputs = [item.sample.output[0].content for item in four_one]
four_one_mini_outputs = [item.sample.output[0].content for item in four_one_mini]

# Create a DataFrame for side-by-side comparison
df = pd.DataFrame({
    "GPT-4.1 Output": four_one_outputs,
    "GPT-4.1-mini Output": four_one_mini_outputs
})

display(df)
```

This DataFrame allows you to compare the responses from both models against your expected answers, helping you assess relevance, correctness, and overall quality.

## Summary and Next Steps

In this guide, you built a workflow to evaluate the web search capabilities of language models using the OpenAI Evals framework.

**Key steps covered:**
1.  Defined a custom dataset for evaluation.
2.  Configured an LLM-based grader for robust assessment.
3.  Set up and ran reproducible evaluations with the latest OpenAI models and the web search tool.
4.  Retrieved and displayed model outputs for inspection.

**Suggested next steps:**
- **Expand the dataset:** Add more diverse and challenging queries to better assess model capabilities across different domains.
- **Analyze results:** Calculate pass/fail rates, visualize performance metrics, or perform error analysis to identify common failure modes.
- **Experiment further:** Test additional models, adjust tool configurations (like `temperature` or `top_p`), or evaluate other information retrieval tasks.
- **Automate reporting:** Generate summary tables, charts, or automated reports for easier sharing and decision-making.

For more detailed information, refer to the [OpenAI Evals documentation](https://platform.openai.com/docs/guides/evals).