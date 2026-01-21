# Fine-Tuning a Text Classifier with OpenAI

This guide walks you through fine-tuning a `babbage-002` model to classify text as either "baseball" or "hockey". You'll start with a raw dataset, prepare it for training, run a fine-tuning job, and evaluate the resulting model.

## Prerequisites

Ensure you have the required Python packages installed and your OpenAI API key configured.

```bash
pip install scikit-learn pandas openai
```

```python
import os
import pandas as pd
import openai
from sklearn.datasets import fetch_20newsgroups

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Step 1: Load and Explore the Dataset

We'll use the 20 Newsgroups dataset, focusing on two sports-related categories.

```python
# Define the categories of interest
categories = ['rec.sport.baseball', 'rec.sport.hockey']

# Load the dataset
sports_dataset = fetch_20newsgroups(
    subset='train',
    shuffle=True,
    random_state=42,
    categories=categories
)
```

Let's examine a sample from the dataset to understand its structure.

```python
# Print the first email
print(sports_dataset['data'][0])
```

The output is a raw email message. Next, check its label.

```python
# Get the category name for the first sample
category_name = sports_dataset.target_names[sports_dataset['target'][0]]
print(f"First sample category: {category_name}")
```

Finally, view the dataset size and class distribution.

```python
# Count total examples and per class
len_all = len(sports_dataset.data)
len_baseball = sum(1 for e in sports_dataset.target if e == 0)
len_hockey = sum(1 for e in sports_dataset.target if e == 1)

print(f"Total examples: {len_all}")
print(f"Baseball examples: {len_baseball}")
print(f"Hockey examples: {len_hockey}")
```

You should see an approximately balanced dataset with 1197 total examples.

## Step 2: Prepare the Data for Fine-Tuning

Transform the raw data into a structured format suitable for OpenAI's fine-tuning API.

```python
# Extract labels and clean text
labels = [sports_dataset.target_names[x].split('.')[-1] for x in sports_dataset['target']]
texts = [text.strip() for text in sports_dataset['data']]

# Create a DataFrame
df = pd.DataFrame(zip(texts, labels), columns=['prompt', 'completion'])
print(df.head())
```

Save the dataset as a JSON Lines file, which is the required format.

```python
df.to_json("sport2.jsonl", orient='records', lines=True)
```

## Step 3: Analyze and Improve the Dataset

Use OpenAI's data preparation tool to analyze the dataset and apply recommended improvements. This step ensures optimal formatting for training.

```bash
openai tools fine_tunes.prepare_data -f sport2.jsonl -q
```

The tool will:
1. Identify the task as classification.
2. Remove overly long examples.
3. Add a suffix separator (`\n\n###\n\n`) to all prompts.
4. Add a leading whitespace to all completions.
5. Split the data into training and validation sets.

After running, you'll have two new files: `sport2_prepared_train.jsonl` and `sport2_prepared_valid.jsonl`.

## Step 4: Launch the Fine-Tuning Job

Upload the prepared files and start the fine-tuning job using the `babbage-002` model.

```python
# Upload training and validation files
train_file = client.files.create(
    file=open("sport2_prepared_train.jsonl", "rb"),
    purpose="fine-tune"
)
valid_file = client.files.create(
    file=open("sport2_prepared_valid.jsonl", "rb"),
    purpose="fine-tune"
)

# Create the fine-tuning job
fine_tuning_job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=valid_file.id,
    model="babbage-002"
)

print(f"Job created with ID: {fine_tuning_job.id}")
```

The job will enter a queue and begin training. You can monitor its progress on the [OpenAI Fine-Tuning Dashboard](https://platform.openai.com/finetune/).

Check the job status programmatically:

```python
# Retrieve job details
job_status = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
print(f"Status: {job_status.status}")
print(f"Finished at: {job_status.finished_at}")
```

## Step 5: Evaluate Model Performance

Once training is complete, download the results to analyze performance metrics.

```python
# Retrieve the results file ID
fine_tune_results = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
result_file_id = fine_tune_results.result_files[0]

# Download the results
result_file = client.files.retrieve(result_file_id)
content = client.files.content(result_file_id)

# Save results locally
with open("result.csv", "wb") as f:
    f.write(content.text.encode("utf-8"))

# Load and display the final accuracy
results = pd.read_csv('result.csv')
final_results = results[results['train_accuracy'].notnull()].tail(1)
print(final_results[['step', 'train_accuracy', 'valid_loss']])
```

The model should achieve very high accuracy (e.g., 99.6%) on the validation set.

## Step 6: Use the Fine-Tuned Model for Predictions

Load the validation set and make predictions with your new model.

```python
# Load validation data
test = pd.read_json('sport2_prepared_valid.jsonl', lines=True)
print(test.head())
```

Define your fine-tuned model ID and make a prediction.

```python
# Get the fine-tuned model name
ft_model = fine_tuning_job.fine_tuned_model

# Prepare the prompt with the required separator
prompt = test['prompt'][0] + '\n\n###\n\n'

# Call the model
response = client.completions.create(
    model=ft_model,
    prompt=prompt,
    max_tokens=1,
    temperature=0
)

prediction = response.choices[0].text.strip()
print(f"Prediction: {prediction}")
print(f"Actual: {test['completion'][0]}")
```

To get detailed prediction probabilities, request log probabilities.

```python
response = client.completions.create(
    model=ft_model,
    prompt=prompt,
    max_tokens=1,
    temperature=0,
    logprobs=2
)

logprobs = response.choices[0].logprobs.top_logprobs
print("Top log probabilities:", logprobs)
```

## Step 7: Test Generalization on New Data

Evaluate how well the model generalizes to unseen text formats, like social media posts.

```python
# Hockey-related tweet
sample_hockey_tweet = """Thank you to the 
@Canes
 and all you amazing Caniacs that have been so supportive! You guys are some of the best fans in the NHL without a doubt! Really excited to start this new chapter in my career with the 
@DetroitRedWings
 !!"""

response = client.completions.create(
    model=ft_model,
    prompt=sample_hockey_tweet + '\n\n###\n\n',
    max_tokens=1,
    temperature=0
)
print(f"Hockey tweet prediction: {response.choices[0].text.strip()}")

# Baseball-related tweet
sample_baseball_tweet = """BREAKING: The Tampa Bay Rays are finalizing a deal to acquire slugger Nelson Cruz from the Minnesota Twins, sources tell ESPN."""

response = client.completions.create(
    model=ft_model,
    prompt=sample_baseball_tweet + '\n\n###\n\n',
    max_tokens=1,
    temperature=0
)
print(f"Baseball tweet prediction: {response.choices[0].text.strip()}")
```

The model should correctly classify both tweets, demonstrating its ability to generalize beyond the email format it was trained on.

## Summary

You have successfully:
1. Loaded and explored a text classification dataset.
2. Prepared the data using OpenAI's analysis tool.
3. Fine-tuned a `babbage-002` model for binary classification.
4. Evaluated the model's performance on a validation set.
5. Used the model to make predictions and tested its generalization.

This fine-tuned classifier is now ready to be integrated into applications requiring sports text classification.