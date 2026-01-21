# Guide: Detecting Prompt Regressions with OpenAI Evals

This guide walks you through using OpenAI Evals to detect if a change to your LLM prompt causes a regression in performance. We'll use a push notification summarizer as our example.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai pydantic
```

```python
import openai
from openai.types.chat import ChatCompletion
import pydantic
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key")
```

## Step 1: Define Your Integration

First, define the data model and the function that uses the LLM to summarize push notifications.

```python
class PushNotifications(pydantic.BaseModel):
    notifications: str

DEVELOPER_PROMPT = """
You are a helpful assistant that summarizes push notifications.
You are given a list of push notifications and you need to collapse them into a single one.
Output only the final summary, nothing else.
"""

def summarize_push_notification(push_notifications: str) -> ChatCompletion:
    result = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": push_notifications},
        ],
    )
    return result
```

Test the function with an example:

```python
example_push_notifications_list = PushNotifications(notifications="""
- Alert: Unauthorized login attempt detected.
- New comment on your blog post: "Great insights!"
- Tonight's dinner recipe: Pasta Primavera.
""")
result = summarize_push_notification(example_push_notifications_list.notifications)
print(result.choices[0].message.content)
```

## Step 2: Set Up the Evaluation

An Eval consists of a **data source configuration** and **testing criteria**. The data source defines the schema of your input data, and the testing criteria define how to judge the LLM's output.

### 2.1 Configure the Data Source

We configure the Eval to expect data matching our `PushNotifications` schema and to include the LLM's output.

```python
data_source_config = {
    "type": "custom",
    "item_schema": PushNotifications.model_json_schema(),
    "include_sample_schema": True,
}
```

This configuration makes two variables available for use in the Eval:
*   `{{item.notifications}}`: The input push notification string.
*   `{{sample.output_text}}`: The LLM-generated summary.

### 2.2 Define the Testing Criteria

We'll use an LLM-as-a-judge (a model grader) to evaluate the summaries. We define the grader's instructions and the template that presents the data.

```python
GRADER_DEVELOPER_PROMPT = """
Label the following push notification summary as either correct or incorrect.
The push notification and the summary will be provided below.
A good push notification summary is concise and snappy.
If it is good, then label it as correct, if not, then incorrect.
"""
GRADER_TEMPLATE_PROMPT = """
Push notifications: {{item.notifications}}
Summary: {{sample.output_text}}
"""

push_notification_grader = {
    "name": "Push Notification Summary Grader",
    "type": "label_model",
    "model": "o3-mini",
    "input": [
        {
            "role": "developer",
            "content": GRADER_DEVELOPER_PROMPT,
        },
        {
            "role": "user",
            "content": GRADER_TEMPLATE_PROMPT,
        },
    ],
    "passing_labels": ["correct"],
    "labels": ["correct", "incorrect"],
}
```

### 2.3 Create the Eval

Now, create the Eval with the configuration and grader.

```python
eval_create_result = openai.evals.create(
    name="Push Notification Summary Workflow",
    metadata={
        "description": "This eval checks if the push notification summary is correct.",
    },
    data_source_config=data_source_config,
    testing_criteria=[push_notification_grader],
)

eval_id = eval_create_result.id
print(f"Created Eval with ID: {eval_id}")
```

## Step 3: Create a Baseline Run

A "Run" is a set of LLM outputs evaluated against the Eval's criteria. First, we create a baseline run using our original, well-behaved prompt.

### 3.1 Prepare Test Data

Define a list of push notification strings to use as test inputs.

```python
push_notification_data = [
    """
- New message from Sarah: "Can you call me later?"
- Your package has been delivered!
- Flash sale: 20% off electronics for the next 2 hours!
""",
    """
- Weather alert: Thunderstorm expected in your area.
- Reminder: Doctor's appointment at 3 PM.
- John liked your photo on Instagram.
""",
    # ... (Include all 10 examples from the original guide)
]
```

### 3.2 Generate and Submit the Baseline Run

Loop through the test data, generate summaries using your original function, and submit them as a Run.

```python
run_data = []
for push_notifications in push_notification_data:
    result = summarize_push_notification(push_notifications)
    run_data.append({
        "item": PushNotifications(notifications=push_notifications).model_dump(),
        "sample": result.model_dump()
    })

eval_run_result = openai.evals.runs.create(
    eval_id=eval_id,
    name="baseline-run",
    data_source={
        "type": "jsonl",
        "source": {
            "type": "file_content",
            "content": run_data,
        }
    },
)
print(f"Baseline Run Report URL: {eval_run_result.report_url}")
```

Visit the report URL to see the performance score for your baseline integration.

## Step 4: Simulate and Detect a Regression

Now, let's simulate a developer accidentally introducing a bad prompt change and see how Evals catches it.

### 4.1 Define the "Bad" Prompt

This prompt instructs the model to be overly verbose.

```python
DEVELOPER_PROMPT_BAD = """
You are a helpful assistant that summarizes push notifications.
You are given a list of push notifications and you need to collapse them into a single one.
You should make the summary longer than it needs to be and include more information than is necessary.
"""

def summarize_push_notification_bad(push_notifications: str) -> ChatCompletion:
    result = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": DEVELOPER_PROMPT_BAD},
            {"role": "user", "content": push_notifications},
        ],
    )
    return result
```

### 4.2 Create a Regression Run

Generate summaries with the bad prompt and submit them as a new Run to the same Eval.

```python
run_data = []
for push_notifications in push_notification_data:
    result = summarize_push_notification_bad(push_notifications)
    run_data.append({
        "item": PushNotifications(notifications=push_notifications).model_dump(),
        "sample": result.model_dump()
    })

eval_run_result = openai.evals.runs.create(
    eval_id=eval_id,
    name="regression-run",
    data_source={
        "type": "jsonl",
        "source": {
            "type": "file_content",
            "content": run_data,
        }
    },
)
print(f"Regression Run Report URL: {eval_run_result.report_url}")
```

When you view this report, you will see a significantly lower score compared to the `baseline-run`. The Eval has successfully flagged the prompt change as a regression.

## Step 5: Using the Responses API (Optional)

The Evals API currently expects the older Completions format. If you are using the newer `responses.create()` API, you can transform the output before submitting a Run.

```python
def summarize_push_notification_responses(push_notifications: str):
    result = openai.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "developer", "content": DEVELOPER_PROMPT},
                    {"role": "user", "content": push_notifications},
                ],
            )
    return result

def transform_response_to_completion(response):
    completion = {
        "model": response.model,
        "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": response.output_text
        },
        "finish_reason": "stop",
    }]
    }
    return completion

run_data = []
for push_notifications in push_notification_data:
    response = summarize_push_notification_responses(push_notifications)
    completion = transform_response_to_completion(response)
    run_data.append({
        "item": PushNotifications(notifications=push_notifications).model_dump(),
        "sample": completion
    })

report_response = openai.evals.runs.create(
    eval_id=eval_id,
    name="responses-run",
    data_source={
        "type": "jsonl",
        "source": {
            "type": "file_content",
            "content": run_data,
        }
    },
)
print(f"Responses API Run Report URL: {report_response.report_url}")
```

## Conclusion

You have successfully set up an OpenAI Eval to monitor the quality of an LLM integration. By establishing a baseline and then testing new prompt changes against it, you can automatically detect regressions before they reach your users. This workflow is essential for maintaining reliable and high-performing AI features.