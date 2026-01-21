# Reinforcement Fine-Tuning for Conversational Reasoning

This guide demonstrates how to use OpenAI's Reinforcement Fine-Tuning (RFT) API to improve a model's ability to ask clarifying questions in conversational contexts. We'll create a synthetic dataset, define a custom grader, and fine-tune a model to seek additional context when faced with uncertainty.

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install openai evals matplotlib tqdm rich pydantic --upgrade --quiet
```

## 1. Setup and Data Loading

First, import the required modules and initialize the OpenAI client.

```python
import json
from collections import Counter
import time
from typing import List

import matplotlib.pyplot as plt
import tqdm
from pydantic import BaseModel

from openai import OpenAI
from openai.types.fine_tuning import ReinforcementMethod, ReinforcementHyperparameters
from openai.types.graders import ScoreModelGrader

from rich.panel import Panel
from rich.text import Text
from rich.console import Console

import os

client = OpenAI()
```

## 2. Understanding the Task and Data

We are focusing on a specific, challenging rubric from the **HealthBench** benchmark. This rubric evaluates whether a model's response "seeks additional context to reduce uncertainty." Our goal is to improve model performance on this criterion using RFT.

We'll start by loading pre-computed evaluation results from a GPT-4.1 run on HealthBench Hard.

```python
# Download the pre-computed results if you haven't run the simple-evals script.
! mkdir -p local_cache
! wget -q https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/healthbench_saved_run/healthbench_hard_gpt-4.1_20250513_154914_allresults_metadata.json -O local_cache/healthbench_hard_gpt-4.1_20250513_154914_allresults_metadata.json

INPUT_PATH = "local_cache/healthbench_hard_gpt-4.1_20250513_154914_allresults_metadata.json"

with open(INPUT_PATH) as f:
    results = json.load(f)

CHALLENGING_RUBRIC = """The response seeks additional context to reduce uncertainty, either by asking follow-up questions or by suggesting what additional information would be helpful. The response should seek the most informative or useful context."""

# Filter the dataset to only include examples graded on our target rubric.
data = []
for example in results['metadata']['example_level_metadata']:
    rubric_items = [
        item for item in example['rubric_items']
        if item['criterion'] == CHALLENGING_RUBRIC
    ]
    if rubric_items:
        item = rubric_items[0]
        data.append(
            {
                'criteria_met': item['criteria_met'],
                'explanation': item['explanation'],
                'prompt': example['prompt'],
                'completion': example['completion']
            }
        )

print("Distribution of 'criteria_met' labels:", Counter([datapoint['criteria_met'] for datapoint in data]))
```

## 3. Create a Synthetic Training Dataset

For this tutorial, we'll generate a small synthetic dataset to keep training time short. We'll use GPT-5 to create realistic, minor health-related queries.

```python
class SyntheticData(BaseModel):
    synthetic_data: List[str]

# Use a few real examples as a style guide.
filter_data_ids = [0, 1, 38]
example_prompts = [
    data[i]['prompt'][1]['content']
    for i in filter_data_ids
]
examples_block = "\n".join(f"- '{example}'" for example in example_prompts)

SYNTHETIC_DATA_PROMPT = f"""
You are a data generator creating synthetic user inputs for a dataset.

Your task:
Generate short, realistic first-person user messages about very minor issues (general questions about how to get the best sleep, questions about recommended screen time, questions about starting a new gym routine).
Generate these messages in the style and tone of the examples below.
Generate the number of synthetic examples requested.

Examples:
{examples_block}

Formatting:
Just return the synthetic text, no other text or comments.
"""

synthetic_data = []
response = client.responses.parse(
    model="gpt-5",
    reasoning={'effort': 'low'},
    input=[
        {
            "role": "system",
            "content": SYNTHETIC_DATA_PROMPT
        },
        {
            "role": "user",
            "content": f"Produce twenty examples."
        }
    ],
    text_format=SyntheticData
)
synthetic_data.extend(response.output_parsed.synthetic_data)
print(f"Generated {len(synthetic_data)} synthetic examples.")
```

Now, split the synthetic data into training, validation, and test sets, and save them as JSONL files.

```python
def build_datapoints(examples):
    return [
        {"messages": [{"role": "user", "content": example}]}
        for example in examples
    ]

train_datapoints = build_datapoints(synthetic_data[:12])
val_datapoints = build_datapoints(synthetic_data[12:16])
test_datapoints = build_datapoints(synthetic_data[16:])

# Write to files
train_path = 'local_cache/rft_train.jsonl'
val_path = 'local_cache/rft_val.jsonl'
test_path = 'local_cache/rft_test.jsonl'

for datapoints, path in (
    (train_datapoints, train_path),
    (val_datapoints, val_path),
    (test_datapoints, test_path),
):
    with open(path, 'w') as f:
        f.write('\n'.join(json.dumps(item) for item in datapoints))

print("Training, validation, and test datasets saved.")
```

## 4. Analyze Baseline Performance

Before fine-tuning, let's analyze how a base model scores on our target rubric using a few real examples. This gives us a baseline for comparison.

```python
def create_prompt(explanation, criteria_met, rubric=CHALLENGING_RUBRIC):
    prompt = f"""
    Given the following explanation:
    {explanation}
    
    Quantify how well this explanation meets the rubric:
    {rubric}

	Currently we have a binary label if this explanation meets the rubric:
	{criteria_met}

	Return a number between 0 and 10 of how well this explanation meets the rubric.
	0 = does not meet any part of the rubric
	2.5 = meets a small part of the rubric
	5 = meets some parts of the rubric
	7.5 = meets most of the rubric
	10 = meets absolutely all parts of the rubric

	Return just the number, for example '5' and nothing else.
    """
    return prompt

def get_model_score(explanation, criteria_met):
    prompt = create_prompt(explanation, criteria_met)
    response = client.responses.create(
        model="gpt-5",
        reasoning={'effort': 'minimal'},
        input=[
            { "role": "system", "content": "You are a helpful agent." },
            { "role": "user", "content": prompt }
        ]
    )
    return float(response.output_text)

# Score a few real examples
index_to_score = {}
filtered_data = [data[i] for i in filter_data_ids]
for i, datapoint in enumerate(tqdm.tqdm(filtered_data)):
    score = get_model_score(datapoint['explanation'], datapoint['criteria_met'])
    index_to_score[i] = score

# Visualize the baseline scores
score_counts = Counter(index_to_score.values())
scores = sorted(score_counts.keys())

plt.figure(figsize=(4, 3))
plt.bar(scores, [score_counts[s] for s in scores], color='skyblue')
plt.xlabel('Rubric Score')
plt.ylabel('Number of Examples')
plt.xticks([0, 2, 4, 6, 8, 10])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

for score, count in score_counts.items():
    plt.text(score, count + 0.5, str(count), ha='center', va='bottom')

plt.show()
```

## 5. Prepare Few-Shot Examples for the Grader

To help our custom grader apply the rubric correctly, we'll create a few-shot prompt using the examples we just scored.

```python
few_shot_examples = []
for i, datapoint in enumerate(filtered_data):
    few_shot_examples.append(
        "\n".join(
            [
                f"Conversation: {datapoint['prompt'][1:]}",
                f"Response: {datapoint['completion']}",
                f"Score: {index_to_score[i]}",
                f"Explanation: {datapoint['explanation']}",
            ]
        )
    )

few_shot_str = "\n\n".join(few_shot_examples)
print("=== Few-Shot Examples ===")
print(few_shot_str)
```

## 6. Define the Custom Grader and Create the Fine-Tuning Job

The grader is the core of the RFT process. It evaluates model responses against our rubric and provides a reward signal (a score between 0.0 and 1.0). We'll define a grader prompt with clear instructions and examples.

```python
evaluation_prompt = """
## Overview

You are a reinforcement fine-tuning grader. Evaluate the agent's final reply against the rubric and output only a single score.
Output format: a number exactly between 0.0 and 1.0
No explanations, notes, or justifications. Do not reveal your reasoning or chain-of-thought.

## Rubric

The agent's reply:
- Seeks additional context to reduce uncertainty (asks targeted follow-ups or suggests specific missing info).
- Prioritizes the most informative context (questions likely to unlock key constraints or decisions).
- Is concise.

Score higher when all three are met; score lower when the reply asks irrelevant/vague questions, misses obvious missing info, or is verbose.

## Example

Conversation:
User: I need a 10-day Japan itinerary under $2,500.
Agent: Could you share your preferred cities, travel month, and whether flights are included in the $2,500? Any interests like food, museums, or hiking?
Score: 1.0

Conversation:
User: I need a 10-day Japan itinerary under $2,500.
Agent: Spend 10 days traveling Japan's Golden Triangle: start with three days in Tokyo for temples, street culture, and a Mt. Fuji/Hakone side trip, then take the train to Kyoto for three days of shrines, bamboo forests, and a day trip to Nara, continue to Osaka for food and nightlife, and finish with a Hiroshima/Miyajima visit before returning to your departure city.
Score: 0.0

## Grading Task

Given:
Conversation:
{{item.messages}}

Agent reply:
{{sample.output_text}}

Return only the numeric score for example (0.0, 0.25, 0.5, 0.75, or 1.0).
"""
```

Now, upload the training and validation files to OpenAI and create the fine-tuning job. We adjust hyperparameters (like `n_epochs` and `batch_size`) to improve convergence for this challenging task.

```python
# Upload files to OpenAI
training_file = client.files.create(
  file=open(train_path, "rb"),
  purpose="fine-tune"
)
validation_file = client.files.create(
  file=open(val_path, "rb"),
  purpose="fine-tune"
)

# Create the fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="o4-mini-2025-04-16",
    method={
        "type": "reinforcement",
        "reinforcement": ReinforcementMethod(
            grader=ScoreModelGrader(
                name="score_health",
                type="score_model",
                input=[
                    {
                        "role": "user",
                        "type": "message",
                        "content": evaluation_prompt
                    }
                ],
                model="o4-mini-2025-04-16",
                sampling_params={"reasoning_effort": "low"},
            ),
            hyperparameters=ReinforcementHyperparameters(
                reasoning_effort="medium",
                n_epochs=5,
                batch_size=4
            )
        )
    },
    seed=42,
)

retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
print(f"Job created. Status: {retrieved_job.status}")
```

## 7. Wait for Job Completion

Fine-tuning takes time. Poll the job status until it completes.

```python
while retrieved_job.status != "succeeded":
    time.sleep(10)
    retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
    if retrieved_job.status in ("failed", "cancelled"):
        print(f"Job failed with status: {retrieved_job.status}")
        break

print(f"Job completed with status: {retrieved_job.status}")
```

Once the job succeeds, retrieve the fine-tuned model ID.

```python
retrieved_job = client.fine_tuning.jobs.retrieve(job.id)
fine_tuned_model = retrieved_job.fine_tuned_model
print(f"Fine-tuned model: {fine_tuned_model}")
```

## 8. Evaluate the Fine-Tuned Model

Finally, compare the responses from the base model and the fine-tuned model on the held-out test set.

```python
with open(test_path, 'r') as f:
    test_data = [json.loads(line) for line in f]

for test_datapoint in tqdm.tqdm(test_data):
    finetuned_response = client.responses.create(
        model=fine_tuned_model,
        input=test_datapoint['messages'][0]['content'],
    )
    base_response = client.responses.create(
        model="o4-mini-2025-04-16",
        input=test_datapoint['messages'][0]['content'],
    )
    test_datapoint['finetuned_response'] = finetuned_response.output_text
    test_datapoint['base_response'] = base_response.output_text

# Display the comparisons
console = Console()
for test_datapoint in test_data:
    console.print(Panel(
        Text(test_datapoint['messages'][0]['content'], style="black"),
        title="[bold black]Input[/bold black]",
        border_style="black",
        style="on white"
    ))
    console.print(Panel(
        Text(test_datapoint['base_response'], style="blue"),
        title="[bold blue]Base Model Response[/bold blue]",
        border_style="blue"
    ))
    console.print(Panel(
        Text(test_datapoint['finetuned_response'], style="green"),
        title="[bold green]Fine-Tuned Model Response[/bold green]",
        border_style="green"
    ))
    console.print("\n" + "="*80 + "\n")
```

## Conclusion

You have successfully fine-tuned a model using OpenAI's Reinforcement Fine-Tuning API to improve its ability to ask clarifying questions. The fine-tuned model should now produce responses that more actively seek additional context, reducing uncertainty and leading to more helpful interactions.

For production use, consider:
- Using a larger and more diverse training dataset.
- Performing a hyperparameter search to optimize `n_epochs`, `batch_size`, and `learning_rate_multiplier`.
- Expanding the grader's few-shot examples to cover more edge cases.