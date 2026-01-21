# Bulk Experimentation with OpenAI Evals: Optimizing a Push Notification Summarizer

This guide walks you through using OpenAI Evals to systematically test multiple prompt and model variations for a push notification summarization task. Evals provide a structured, task-oriented framework to measure and improve your LLM integration's performance.

## Prerequisites

Ensure you have the OpenAI Python package installed and your API key configured.

```bash
pip install openai pydantic
```

```python
import os
import pydantic
import openai
from openai.types.chat import ChatCompletion

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key")
```

## 1. Define Your Use Case and Integration

First, define the data structure for your push notifications and the core summarization function you want to evaluate.

```python
class PushNotifications(pydantic.BaseModel):
    notifications: str

DEVELOPER_PROMPT = """
You are a helpful assistant that summarizes push notifications.
You are given a list of push notifications and you need to collapse them into a single one.
Output only the final summary, nothing else.
"""

def summarize_push_notification(push_notifications: str) -> ChatCompletion:
    """Calls the OpenAI API to summarize a list of push notifications."""
    result = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": push_notifications},
        ],
    )
    return result

# Quick test
example_notifications = PushNotifications(notifications="""
- Alert: Unauthorized login attempt detected.
- New comment on your blog post: "Great insights!"
- Tonight's dinner recipe: Pasta Primavera.
""")
result = summarize_push_notification(example_notifications.notifications)
print(result.choices[0].message.content)
```

## 2. Configure the Evaluation (Eval)

An Eval defines the shared configuration for all your test runs. It has two key components:
1.  **Data Source Configuration (`data_source_config`)**: Defines the schema (variables) available for testing.
2.  **Testing Criteria (`testing_criteria`)**: Defines how to judge the quality of each result.

### 2.1. Set Up the Data Source

Configure the Eval to expect your `PushNotifications` schema and to include the model's generated output.

```python
data_source_config = {
    "type": "custom",
    "item_schema": PushNotifications.model_json_schema(),
    "include_sample_schema": True,  # Enables the {{sample.output_text}} variable
}
```

This configuration makes two variables available in your eval:
*   `{{item.notifications}}`: The input list of notifications.
*   `{{sample.output_text}}`: The generated summary from your model.

### 2.2. Define the Testing Criteria (Grader)

Create an LLM-as-a-judge grader to categorize each summary. This grader will compare the original notifications against the generated summary.

```python
GRADER_DEVELOPER_PROMPT = """
Categorize the following push notification summary into the following categories:
1. concise-and-snappy
2. drops-important-information
3. verbose
4. unclear
5. obscures-meaning
6. other

You'll be given the original list of push notifications and the summary like this:

<push_notifications>
...notificationlist...
</push_notifications>
<summary>
...summary...
</summary>

You should only pick one of the categories above, pick the one which most closely matches and why.
"""

GRADER_TEMPLATE_PROMPT = """
<push_notifications>{{item.notifications}}</push_notifications>
<summary>{{sample.output_text}}</summary>
"""

push_notification_grader = {
    "name": "Push Notification Summary Grader",
    "type": "label_model",
    "model": "o3-mini",
    "input": [
        {"role": "developer", "content": GRADER_DEVELOPER_PROMPT},
        {"role": "user", "content": GRADER_TEMPLATE_PROMPT},
    ],
    "passing_labels": ["concise-and-snappy"],  # Only this label is considered a "pass"
    "labels": [
        "concise-and-snappy",
        "drops-important-information",
        "verbose",
        "unclear",
        "obscures-meaning",
        "other",
    ],
}
```

### 2.3. Create the Eval

Combine the data source and testing criteria to create your Eval.

```python
eval_create_result = openai.evals.create(
    name="Push Notification Bulk Experimentation Eval",
    metadata={
        "description": "This eval tests many prompts and models to find the best performing combination.",
    },
    data_source_config=data_source_config,
    testing_criteria=[push_notification_grader],
)
eval_id = eval_create_result.id
print(f"Created Eval with ID: {eval_id}")
```

## 3. Prepare Test Data and Prompts

Now, prepare the data you'll test against and the different prompt variations you want to evaluate.

### 3.1. Define Test Data

Create a list of realistic push notification bundles.

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
    # ... (Include all 10 notification bundles from the original example)
]
```

### 3.2. Define Prompt Variations

Create three prompt variations of increasing complexity to test.

```python
PROMPT_PREFIX = """
You are a helpful assistant that takes in an array of push notifications and returns a collapsed summary of them.
The push notification will be provided as follows:
<push_notifications>
...notificationlist...
</push_notifications>

You should return just the summary and nothing else.
"""

PROMPT_VARIATION_BASIC = f"""
{PROMPT_PREFIX}

You should return a summary that is concise and snappy.
"""

PROMPT_VARIATION_WITH_EXAMPLES = f"""
{PROMPT_VARIATION_BASIC}

Here is an example of a good summary:
<push_notifications>
- Traffic alert: Accident reported on Main Street.- Package out for delivery: Expected by 5 PM.- New friend suggestion: Connect with Emma.
</push_notifications>
<summary>
Traffic alert, package expected by 5pm, suggestion for new friend (Emily).
</summary>
"""

PROMPT_VARIATION_WITH_NEGATIVE_EXAMPLES = f"""
{PROMPT_VARIATION_WITH_EXAMPLES}

Here is an example of a bad summary:
<push_notifications>
- Traffic alert: Accident reported on Main Street.- Package out for delivery: Expected by 5 PM.- New friend suggestion: Connect with Emma.
</push_notifications>
<summary>
Traffic alert reported on main street. You have a package that will arrive by 5pm, Emily is a new friend suggested for you.
</summary>
"""

prompts = [
    ("basic", PROMPT_VARIATION_BASIC),
    ("with_examples", PROMPT_VARIATION_WITH_EXAMPLES),
    ("with_negative_examples", PROMPT_VARIATION_WITH_NEGATIVE_EXAMPLES),
]

models = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
```

## 4. Execute Bulk Test Runs

Finally, loop through all combinations of prompts and models to create individual test runs. The `completions` data source type tells the Evals system to automatically call the OpenAI API for each item.

```python
for prompt_name, prompt in prompts:
    for model in models:
        # Configure the data source for this specific run
        run_data_source = {
            "type": "completions",
            "input_messages": {
                "type": "template",
                "template": [
                    {"role": "developer", "content": prompt},
                    {
                        "role": "user",
                        "content": "<push_notifications>{{item.notifications}}</push_notifications>",
                    },
                ],
            },
            "model": model,
            "source": {
                "type": "file_content",
                "content": [
                    {"item": PushNotifications(notifications=notification).model_dump()}
                    for notification in push_notification_data
                ],
            },
        }

        # Create the run
        run_create_result = openai.evals.runs.create(
            eval_id=eval_id,
            name=f"bulk_{prompt_name}_{model}",
            data_source=run_data_source,
        )
        print(f"Report URL for {model}, {prompt_name}: {run_create_result.report_url}")
```

## Next Steps

Congratulations! You have just tested 9 different configurations (3 prompts × 3 models) across your dataset. You can now:

1.  **View Reports**: Click the provided report URLs to see detailed results for each run.
2.  **Analyze Performance**: Compare which prompt and model combination yields the highest percentage of "concise-and-snappy" summaries.
3.  **Iterate**: Use these insights to refine your prompts further or test additional models.

This structured approach allows you to move from guesswork to data-driven optimization of your LLM integrations.