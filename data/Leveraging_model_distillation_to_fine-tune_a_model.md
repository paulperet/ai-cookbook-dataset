# Fine-Tuning a Model via Distillation: A Wine Classification Tutorial

## Introduction

OpenAI's **Distillation** feature enables you to use the outputs of a large, powerful model (like `gpt-4o`) to fine-tune a smaller, more efficient model (like `gpt-4o-mini`). This process can significantly reduce inference costs and latency for specific tasks while preserving performance.

In this tutorial, you will build a wine grape variety classifier. You will:
1.  Prepare a dataset of French wine reviews.
2.  Use **Structured Outputs** to ensure model predictions are constrained to a valid list of grape varieties.
3.  Generate predictions from both `gpt-4o` and `gpt-4o-mini` to establish a performance baseline.
4.  Distill the knowledge from `gpt-4o` to create a fine-tuned version of `gpt-4o-mini`.
5.  Evaluate the distilled model, demonstrating a significant performance improvement over the base `gpt-4o-mini`.

## Prerequisites

Ensure your OpenAI API key is set in your environment as `OPENAI_API_KEY`. Then, install and import the required libraries.

```bash
pip install openai tiktoken numpy pandas tqdm --quiet
```

```python
import openai
import json
import tiktoken
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import concurrent.futures
import pandas as pd

# Initialize the OpenAI client
client = OpenAI()
```

## Step 1: Load and Prepare the Dataset

You will use a wine reviews dataset from Kaggle. For this tutorial, you'll filter it to French wines and focus on grape varieties with sufficient examples.

### 1.1 Load and Filter the Data

```python
# Load the dataset
df = pd.read_csv('data/winemag/winemag-data-130k-v2.csv')

# Filter for French wines
df_france = df[df['country'] == 'France']

# Identify and remove rare grape varieties (fewer than 5 occurrences)
varieties_less_than_five_list = df_france['variety'].value_counts()[df_france['variety'].value_counts() < 5].index.tolist()
df_france = df_france[~df_france['variety'].isin(varieties_less_than_five_list)]

# Create a manageable subset for initial processing
df_france_subset = df_france.sample(n=500)
```

### 1.2 Extract Target Classes

You need the complete list of grape varieties for your structured output schema and prompts.

```python
# Get the unique list of grape varieties
varieties = np.array(df_france['variety'].unique()).astype('str')
print(f"Number of target grape varieties: {len(varieties)}")
```

## Step 2: Construct the Classification Prompt

You will create a function that builds a detailed prompt for the model using data from a single wine review.

```python
def generate_prompt(row, varieties):
    """Creates a prompt asking the model to predict the grape variety."""
    variety_list = ', '.join(varieties)
    
    prompt = f"""
    Based on this wine review, guess the grape variety:
    This wine is produced by {row['winery']} in the {row['province']} region of {row['country']}.
    It was grown in {row['region_1']}. It is described as: "{row['description']}".
    The wine has been reviewed by {row['taster_name']} and received {row['points']} points.
    The price is {row['price']}.

    Here is a list of possible grape varieties to choose from: {variety_list}.
    
    What is the likely grape variety? Answer only with the grape variety name or blend from the list.
    """
    return prompt

# Test the prompt function
example_prompt = generate_prompt(df_france.iloc[0], varieties)
print(example_prompt[:300] + "...")  # Print first 300 chars
```

## Step 3: Estimate Token Usage and Cost

Before running many API calls, it's good practice to estimate the token count and associated cost.

```python
# Load the tokenizer for GPT-4
enc = tiktoken.encoding_for_model("gpt-4o")
total_tokens = 0

# Count tokens for all prompts in the subset
for index, row in df_france_subset.iterrows():
    prompt = generate_prompt(row, varieties)
    total_tokens += len(enc.encode(prompt))

print(f"Total tokens in dataset: {total_tokens}")
print(f"Number of prompts: {len(df_france_subset)}")

# Calculate estimated inference cost (prices as of 2024-10-16)
gpt4o_token_price = 2.50 / 1_000_000  # $2.50 per 1M tokens
gpt4o_mini_token_price = 0.150 / 1_000_000  # $0.15 per 1M tokens

total_gpt4o_cost = gpt4o_token_price * total_tokens
total_gpt4o_mini_cost = gpt4o_mini_token_price * total_tokens

print(f"Estimated cost for gpt-4o: ${total_gpt4o_cost:.4f}")
print(f"Estimated cost for gpt-4o-mini: ${total_gpt4o_mini_cost:.4f}")
```

## Step 4: Define the Structured Output Schema

Using Structured Outputs ensures the model's predictions are valid grape varieties from your list and returned in a consistent JSON format. This is crucial for reliable evaluation and distillation.

```python
# Define the JSON schema for the response
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "grape-variety",
        "schema": {
            "type": "object",
            "properties": {
                "variety": {
                    "type": "string",
                    "enum": varieties.tolist()  # Constrain output to this list
                }
            },
            "additionalProperties": False,
            "required": ["variety"],
        },
        "strict": True
    }
}
```

## Step 5: Create a Function to Call the API

This function calls the Chat Completions API for a given model and prompt. Crucially, it uses the `store=True` parameter to save the completion for later use in distillation.

```python
# A metadata tag to identify all completions from this experiment
metadata_value = "wine-distillation"

def call_model(model, prompt):
    """Calls the OpenAI API, stores the completion, and returns the parsed variety."""
    response = client.chat.completions.create(
        model=model,
        store=True,  # Store the completion for distillation
        metadata={"distillation": metadata_value},
        messages=[
            {
                "role": "system",
                "content": "You're a sommelier expert and you know everything about wine. You answer precisely with the name of the variety/blend."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=response_format
    )
    # Parse the structured JSON response
    return json.loads(response.choices[0].message.content.strip())['variety']

# Test the function
test_answer = call_model('gpt-4o', generate_prompt(df_france_subset.iloc[0], varieties))
print(f"Test prediction: {test_answer}")
```

## Step 6: Process the Dataset in Parallel

To generate predictions for the entire dataset efficiently, you'll use parallel processing.

```python
def process_example(index, row, model, df, progress_bar):
    """Processes a single row: generates a prompt, calls the model, and stores the result."""
    try:
        prompt = generate_prompt(row, varieties)
        df.at[index, model + "-variety"] = call_model(model, prompt)
        progress_bar.update(1)
    except Exception as e:
        print(f"Error processing model {model} for row {index}: {str(e)}")

def process_dataframe(df, model):
    """Processes an entire DataFrame for a given model using parallel threads."""
    with tqdm(total=len(df), desc=f"Processing {model}") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_example, index, row, model, df, progress_bar): index
                for index, row in df.iterrows()
            }
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in concurrent task: {str(e)}")
    return df
```

## Step 7: Generate Baseline Predictions

Now, run the subset through both `gpt-4o` and `gpt-4o-mini` to establish their baseline accuracy.

```python
# Get predictions from gpt-4o (the "teacher" model)
print("Generating predictions with gpt-4o...")
df_france_subset = process_dataframe(df_france_subset, "gpt-4o")

# Get predictions from gpt-4o-mini (the "student" model)
print("\nGenerating predictions with gpt-4o-mini...")
df_france_subset = process_dataframe(df_france_subset, "gpt-4o-mini")
```

## Step 8: Evaluate Baseline Model Performance

Compare the predictions from each model against the true grape varieties.

```python
def get_accuracy(model, df):
    """Calculates the accuracy of a model's predictions."""
    return np.mean(df['variety'] == df[model + '-variety'])

models = ['gpt-4o', 'gpt-4o-mini']
print("Baseline Model Accuracy:")
for model in models:
    accuracy = get_accuracy(model, df_france_subset) * 100
    print(f"  {model}: {accuracy:.2f}%")
```

You should see that `gpt-4o` is significantly more accurate than `gpt-4o-mini` on this task. The goal of distillation is to transfer this superior performance to the smaller model.

## Step 9: Distill the Teacher Model

You will now use the stored completions from `gpt-4o` to fine-tune `gpt-4o-mini`.

1.  **Navigate to Stored Completions:** Go to the [OpenAI Chat Completions page](https://platform.openai.com/chat-completions).
2.  **Filter Completions:** Select the model `gpt-4o` and filter by the metadata tag `distillation: wine-distillation`. This isolates the 500 completions you just generated.
3.  **Initiate Distillation:** Click the **"Distill"** button in the top-right corner.
4.  **Configure the Job:** In the distillation interface:
    *   Select `gpt-4o-mini` as the base model.
    *   Review the default training parameters (you can adjust these if desired).
    *   Start the fine-tuning job.
5.  **Retrieve the Job ID:** Once the job starts, copy its ID from the fine-tuning dashboard. You will use this to monitor progress and retrieve your new model.

## Step 10: Monitor the Fine-Tuning Job

Use the Python client to check the status of your distillation job and get the name of your new fine-tuned model.

```python
# Replace 'ftjob-...' with your actual fine-tuning job ID
finetune_job_id = "ftjob-pRyNWzUItmHpxmJ1TX7FOaWe"
finetune_job = client.fine_tuning.jobs.retrieve(finetune_job_id)

print(f"Job Status: {finetune_job.status}")

if finetune_job.status == 'succeeded':
    fine_tuned_model = finetune_job.fine_tuned_model
    print(f'Fine-tuned model name: {fine_tuned_model}')
    # Add the new model to your list for evaluation
    models.append(fine_tuned_model)
else:
    print("Job is still running. Please wait for it to succeed.")
```

## Step 11: Evaluate the Distilled Model

Create a new, separate validation dataset to test all models, including your newly fine-tuned one.

```python
# Create a fresh validation set
validation_dataset = df_france.sample(n=300).copy()

# Run predictions for all models on the validation set
print("Running predictions on validation set...")
for model in models:
    print(f"  Processing {model}...")
    validation_dataset = process_dataframe(validation_dataset, model)

# Calculate and compare final accuracy
print("\nFinal Model Accuracy on Validation Set:")
for model in models:
    accuracy = get_accuracy(model, validation_dataset) * 100
    print(f"  {model}: {accuracy:.2f}%")
```

## Conclusion

You have successfully implemented a model distillation workflow. The results should show that your fine-tuned `gpt-4o-mini` model achieves accuracy much closer to `gpt-4o` and significantly higher than the base `gpt-4o-mini`. This demonstrates the power of distillation: you can now use this smaller, cheaper, and faster model for your specific wine classification task while maintaining high performance.

**Key Takeaways:**
*   **Structured Outputs** are essential for ensuring clean, consistent, and valid model predictions, which simplifies evaluation and distillation.
*   **Distillation** allows you to capture the task-specific knowledge of a large model and transfer it to a smaller one.
*   The fine-tuned model offers a compelling trade-off: it retains the cost and latency benefits of `gpt-4o-mini` while delivering accuracy superior to its base version.

You can now deploy this fine-tuned model via the OpenAI API for efficient, high-quality grape variety predictions.