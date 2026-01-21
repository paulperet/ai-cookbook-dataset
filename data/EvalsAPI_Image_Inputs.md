# Evaluating Image-Based AI Responses with OpenAI Evals

This guide demonstrates how to use OpenAI's Evals framework to evaluate model performance on image-based tasks. You will learn to set up an evaluation that:
1. **Generates model responses** to prompts about images.
2. **Scores those responses** using an LLM-as-a-judge (model grader) against a reference answer.

We'll use a sample dataset to create a custom evaluation, run it, and analyze the results.

## Prerequisites & Setup

First, install the required packages and import the necessary libraries.

```bash
pip install openai datasets pandas --quiet
```

```python
# Import libraries
from datasets import load_dataset
from openai import OpenAI
import os
import json
import time
import pandas as pd
```

Ensure your OpenAI API key is set in your environment variables.

```python
# The client will automatically use the OPENAI_API_KEY environment variable
client = OpenAI()
```

## Step 1: Prepare Your Dataset

We'll use the **VibeEval** dataset from Hugging Face, which contains prompts, images, and reference answers.

```python
# Load the dataset
dataset = load_dataset("RekaAI/VibeEval")
```

Next, we format the data for the Evals API. The API expects a list of items, each containing an image URL, a prompt, and a reference answer.

```python
# Create the data source list
evals_data_source = []

# Select the first 3 examples for this tutorial
for example in dataset["test"].select(range(3)):
    evals_data_source.append({
        "item": {
            "media_url": example["media_url"],  # Image web URL
            "reference": example["reference"],   # Reference answer
            "prompt": example["prompt"]          # User prompt
        }
    })
```

Each item in `evals_data_source` will have the following structure:

```json
{
  "item": {
    "media_url": "https://storage.googleapis.com/.../pizza.jpg",
    "reference": "This appears to be a classic Margherita pizza...",
    "prompt": "What ingredients do I need to make this?"
  }
}
```

## Step 2: Configure the Evaluation

An evaluation consists of two main parts: a **Data Source Configuration** and a **Testing Criteria (Grader)**.

### 2.1 Define the Data Source Schema

This schema tells the Evals API the structure of your data.

```python
data_source_config = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
          "media_url": { "type": "string" },
          "reference": { "type": "string" },
          "prompt": { "type": "string" }
        },
        "required": ["media_url", "reference", "prompt"]
      },
    "include_sample_schema": True,  # Enables the sampling step
}
```

### 2.2 Create the Model Grader

The grader is an LLM that judges the model's response. It receives the image, the original prompt, the reference answer, and the model's generated response, then outputs a score between 0 and 1.

**Important:** The image URL must be placed within an `input_image` object. If passed as plain text, it will be interpreted as a string.

```python
grader_config = {
    "type": "score_model",
    "name": "Score Model Grader",
    "input":[
        {
            "role": "system",
            "content": "You are an expert grader. Judge how well the model response suits the image and prompt as well as matches the meaning of the reference answer. Output a score of 1 if great. If it's somewhat compatible, output a score around 0.5. Otherwise, give a score of 0."
        },
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "Prompt: {{ item.prompt }}."},
                { "type": "input_image", "image_url": "{{ item.media_url }}", "detail": "auto" },
                { "type": "input_text", "text": "Reference answer: {{ item.reference }}. Model response: {{ sample.output_text }}."}
            ]
        }
    ],
    "pass_threshold": 0.9,
    "range": [0, 1],
    "model": "o4-mini"  # Ensure this model supports image inputs
}
```

### 2.3 Create the Eval Object

Now, we create the evaluation definition using the client.

```python
eval_object = client.evals.create(
    name="Image Grading",
    data_source_config=data_source_config,
    testing_criteria=[grader_config],
)
print(f"Created eval with ID: {eval_object.id}")
```

## Step 3: Run the Evaluation

An evaluation run performs two actions:
1. **Sampling:** Uses a model to generate responses for each item in your data source.
2. **Grading:** Uses the grader you defined to score each generated response.

First, define the sampling prompt template. This is the instruction sent to the model to generate a response.

```python
sampling_messages = [
    {
        "role": "user",
        "type": "message",
        "content": {
            "type": "input_text",
            "text": "{{ item.prompt }}"
        }
    },
    {
        "role": "user",
        "type": "message",
        "content": {
            "type": "input_image",
            "image_url": "{{ item.media_url }}",
            "detail": "auto"
        }
    }
]
```

Now, start the evaluation run.

```python
eval_run = client.evals.runs.create(
    name="Image Input Eval Run",
    eval_id=eval_object.id,
    data_source={
        "type": "responses",  # Sample using the Responses API
        "source": {
            "type": "file_content",
            "content": evals_data_source
        },
        "model": "gpt-4o-mini",  # Model for generating responses
        "input_messages": {
            "type": "template",
            "template": sampling_messages
        }
    }
)
print(f"Started eval run with ID: {eval_run.id}")
```

## Step 4: Retrieve and Analyze Results

The run executes asynchronously. Poll its status until it completes.

```python
while True:
    run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
    if run.status == "completed" or run.status == "failed":
        print(f"Run finished with status: {run.status}")
        break
    time.sleep(5)  # Wait 5 seconds before checking again
```

Once complete, fetch the output items and display them in a clean DataFrame.

```python
# Get all output items from the run
output_items = list(client.evals.runs.output_items.list(
    run_id=run.id, eval_id=eval_object.id
))

# Create a summary DataFrame
df = pd.DataFrame({
    "prompt": [item.datasource_item["prompt"] for item in output_items],
    "reference": [item.datasource_item["reference"] for item in output_items],
    "model_response": [item.sample.output[0].content for item in output_items],
    "grading_results": [item.results[0]["sample"]["output"][0]["content"]
                        for item in output_items]
})

print(df.to_string())
```

**Example Output:**

| prompt | reference | model_response | grading_results |
|--------|-----------|----------------|-----------------|
| Please provide latex code to replicate this table | Below is the latex code for your table:\n```te... | Certainly! Below is the LaTeX code to replicat... | {"steps":[{"description":"Assess if the provid... |
| What ingredients do I need to make this? | This appears to be a classic Margherita pizza,... | To make a classic Margherita pizza like the on... | {"steps":[{"description":"Check if model ident... |
| Is this safe for a vegan to eat? | Based on the image, this dish appears to be a ... | To determine if the dish is safe for a vegan t... | {"steps":[{"description":"Compare model respon... |

### 4.1 Inspect a Detailed Output Item

To understand the grader's reasoning, you can inspect the full JSON structure of any output item.

```python
# Examine the first item in detail
first_item = output_items[0]
print(json.dumps(dict(first_item), indent=2, default=str))
```

The output is a comprehensive JSON object containing:
- The original data source item (prompt, image URL, reference).
- The model's generated response (`sample`).
- The grader's detailed evaluation (`results`), including a step-by-step reasoning chain and a final score.

## Summary

You have successfully created and run an image-based evaluation using the OpenAI Evals API. This workflow allows you to systematically assess how well a model interprets images and responds to prompts by comparing its outputs to reference answers.

**Key Takeaways:**
1.  **Data Preparation:** Structure your data with clear image URLs, prompts, and reference answers.
2.  **Grader Design:** Craft a precise grader prompt that instructs the LLM judge on how to score responses. Remember to format image inputs correctly.
3.  **Asynchronous Execution:** Evaluation runs are async; poll for status until completion.
4.  **Result Analysis:** Results provide both a pass/fail status and detailed grader reasoning for interpretability.

You can extend this pattern to evaluate other multimodal tasks by adjusting the data source schema and the grader's instructions. For more advanced use cases, such as using pre-generated model logs as a data source, refer to the [OpenAI Evals API documentation](https://platform.openai.com/docs/evals/overview).