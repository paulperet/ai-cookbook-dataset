# Guide: Monitoring Prompt Changes with Evals

This guide demonstrates how to use OpenAI Evals to detect regressions in your LLM prompts. We'll create a monitoring system for a push notification summarizer, comparing two prompt versions to identify performance issues.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai
```

```python
import os
from openai import AsyncOpenAI
import asyncio

# Set your API key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key")
client = AsyncOpenAI()
```

## Step 1: Generate Test Data

First, create simulated push notification data to test your summarizer.

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
    # ... (additional notification sets)
]
```

## Step 2: Define Your Prompt Versions

Create two prompt versions: one "good" (concise) and one "bad" (verbose) to simulate a regression.

```python
PROMPTS = [
    (
        """
        You are a helpful assistant that summarizes push notifications.
        You are given a list of push notifications and you need to collapse them into a single one.
        Output only the final summary, nothing else.
        """,
        "v1"
    ),
    (
        """
        You are a helpful assistant that summarizes push notifications.
        You are given a list of push notifications and you need to collapse them into a single one.
        The summary should be longer than it needs to be and include more information than is necessary.
        Output only the final summary, nothing else.
        """,
        "v2"
    )
]
```

## Step 3: Generate and Store Completions

Make chat completion calls for each notification set with both prompt versions, enabling logging with `store=True`.

```python
tasks = []
for notifications in push_notification_data:
    for (prompt, version) in PROMPTS:
        tasks.append(client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": notifications},
            ],
            store=True,
            metadata={"prompt_version": version, "usecase": "push_notifications_summarizer"},
        ))

# Execute all completion requests
await asyncio.gather(*tasks)
```

**Important**: Verify your completions appear at [https://platform.openai.com/logs](https://platform.openai.com/logs) before proceeding.

## Step 4: Configure Your Eval Data Source

Define the data source configuration to filter your stored completions by metadata.

```python
data_source_config = {
    "type": "stored_completions",
    "metadata": {
        "usecase": "push_notifications_summarizer"
    }
}
```

This configuration makes two variables available for your evaluation:
- `{{item.input}}`: The messages sent to the completion call
- `{{sample.output_text}}`: The assistant's text response

## Step 5: Define Testing Criteria

Create a model grader (LLM-as-a-judge) to evaluate summary quality.

```python
GRADER_DEVELOPER_PROMPT = """
Label the following push notification summary as either correct or incorrect.
The push notification and the summary will be provided below.
A good push notification summary is concise and snappy.
If it is good, then label it as correct, if not, then incorrect.
"""

GRADER_TEMPLATE_PROMPT = """
Push notifications: {{item.input}}
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

## Step 6: Create Your Eval

Establish the eval with your data source configuration and testing criteria.

```python
eval_create_result = await client.evals.create(
    name="Push Notification Completion Monitoring",
    metadata={"description": "This eval monitors completions"},
    data_source_config=data_source_config,
    testing_criteria=[push_notification_grader],
)

eval_id = eval_create_result.id
```

## Step 7: Create Evaluation Runs

Now create separate runs to compare performance between your two prompt versions.

### Run 1: Evaluate Prompt Version v1

```python
eval_run_result = await client.evals.runs.create(
    eval_id=eval_id,
    name="v1-run",
    data_source={
        "type": "completions",
        "source": {
            "type": "stored_completions",
            "metadata": {
                "prompt_version": "v1",
            }
        }
    }
)
print(f"View report: {eval_run_result.report_url}")
```

### Run 2: Evaluate Prompt Version v2

```python
eval_run_result_v2 = await client.evals.runs.create(
    eval_id=eval_id,
    name="v2-run",
    data_source={
        "type": "completions",
        "source": {
            "type": "stored_completions",
            "metadata": {
                "prompt_version": "v2",
            }
        }
    }
)
print(f"View report: {eval_run_result_v2.report_url}")
```

## Step 8: Test with a Different Model

You can also test how your prompts perform with a different model (like GPT-4o) by generating new completions during the eval run.

```python
tasks = []
for prompt_version in ["v1", "v2"]:
    tasks.append(client.evals.runs.create(
        eval_id=eval_id,
        name=f"post-fix-new-model-run-{prompt_version}",
        data_source={
            "type": "completions",
            "input_messages": {
                "type": "item_reference",
                "item_reference": "item.input",
            },
            "model": "gpt-4o",
            "source": {
                "type": "stored_completions",
                "metadata": {
                    "prompt_version": prompt_version,
                }
            }
        },
    ))

result = await asyncio.gather(*tasks)
for run in result:
    print(f"View report: {run.report_url}")
```

## Step 9: Analyze Results

Visit the report URLs to compare performance between prompt versions. You'll likely discover that:
- Prompt version v1 performs well with concise summaries
- Prompt version v2 shows regressions with overly verbose summaries

## Conclusion

You've successfully set up an evaluation pipeline to monitor prompt changes. When you detect regressions (like the verbose summaries from v2), you can:
1. Revert to the previous prompt version
2. Make further prompt improvements
3. Continue monitoring with additional eval runs

This iterative process helps maintain and improve your LLM integration's performance over time.