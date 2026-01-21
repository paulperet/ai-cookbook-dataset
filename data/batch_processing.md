# Batch Processing with the OpenAI Batch API: A Practical Guide

The new Batch API enables you to create asynchronous batch jobs at a lower cost and with higher rate limits. Batches are completed within 24 hours, often sooner depending on global usage.

Ideal use cases include:
- Tagging, captioning, or enriching content
- Categorizing support tickets
- Performing sentiment analysis on customer feedback
- Generating summaries or translations for document collections

This guide walks you through two practical examples:
1. **Categorizing movies** using `gpt-4o-mini` with JSON mode.
2. **Captioning images** of furniture items using vision capabilities.

## Prerequisites

Ensure you have the latest OpenAI SDK installed and import the necessary libraries.

```bash
pip install openai --upgrade pandas
```

```python
import json
from openai import OpenAI
import pandas as pd
```

Initialize the OpenAI client:

```python
client = OpenAI()
```

## Example 1: Categorizing Movies

In this example, you will use `gpt-4o-mini` to extract movie categories and a one-sentence summary from movie descriptions, outputting structured JSON.

### Step 1: Load the Dataset

Use the IMDB top 1000 movies dataset.

```python
dataset_path = "data/imdb_top_1000.csv"
df = pd.read_csv(dataset_path)
df.head()
```

### Step 2: Define the Categorization Function

Create a system prompt and a function to test the categorization using the Chat Completions endpoint.

```python
categorize_system_prompt = '''
Your goal is to extract movie categories from movie descriptions, as well as a 1-sentence summary for these movies.
You will be provided with a movie description, and you will output a json object containing the following information:

{
    categories: string[] // Array of categories based on the movie description,
    summary: string // 1-sentence summary of the movie based on the movie description
}

Categories refer to the genre or type of the movie, like "action", "romance", "comedy", etc. Keep category names simple and use only lower case letters.
Movies can have several categories, but try to keep it under 3-4. Only mention the categories that are the most obvious based on the description.
'''

def get_categories(description):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={ "type": "json_object" },
        messages=[
            { "role": "system", "content": categorize_system_prompt },
            { "role": "user", "content": description }
        ]
    )
    return response.choices[0].message.content
```

Test the function on a few examples:

```python
for _, row in df[:5].iterrows():
    description = row['Overview']
    title = row['Series_Title']
    result = get_categories(description)
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
```

### Step 3: Prepare the Batch File

The batch file must be in `jsonl` format, with each line representing a request. Each request includes a unique `custom_id`.

```python
tasks = []
for index, row in df.iterrows():
    description = row['Overview']
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": categorize_system_prompt },
                { "role": "user", "content": description }
            ]
        }
    }
    tasks.append(task)
```

Write the tasks to a JSONL file:

```python
file_name = "data/batch_tasks_movies.jsonl"
with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')
```

### Step 4: Upload the File and Create a Batch Job

Upload the file and create a batch job with a 24-hour completion window.

```python
batch_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
)

batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

### Step 5: Monitor the Batch Job

Check the job status until it is `completed`.

```python
batch_job = client.batches.retrieve(batch_job.id)
print(batch_job)
```

### Step 6: Retrieve and Process Results

Once the job is complete, download the results file and load the data.

```python
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content

result_file_name = "data/batch_job_results_movies.jsonl"
with open(result_file_name, 'wb') as file:
    file.write(result)

results = []
with open(result_file_name, 'r') as file:
    for line in file:
        json_object = json.loads(line.strip())
        results.append(json_object)
```

### Step 7: Match Results to Inputs

Results are not returned in order. Use the `custom_id` to match each result to the original input.

```python
for res in results[:5]:
    task_id = res['custom_id']
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    movie = df.iloc[int(index)]
    description = movie['Overview']
    title = movie['Series_Title']
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
```

## Example 2: Captioning Images

In this example, you will use `gpt-4o-mini` to generate descriptive captions for images of furniture items.

### Step 1: Load the Dataset

Use the Amazon furniture dataset.

```python
dataset_path = "data/amazon_furniture_dataset.csv"
df = pd.read_csv(dataset_path)
df.head()
```

### Step 2: Define the Captioning Function

Create a system prompt and a function to test image captioning.

```python
caption_system_prompt = '''
Your goal is to generate short, descriptive captions for images of items.
You will be provided with an item image and the name of that item and you will output a caption that captures the most important information about the item.
If there are multiple items depicted, refer to the name provided to understand which item you should describe.
Your generated caption should be short (1 sentence), and include only the most important information about the item.
The most important information could be: the type of item, the style (if mentioned), the material or color if especially relevant and/or any distinctive features.
Keep it short and to the point.
'''

def get_caption(img_url, title):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=300,
        messages=[
            { "role": "system", "content": caption_system_prompt },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": title },
                    { "type": "image_url", "image_url": { "url": img_url } }
                ]
            }
        ]
    )
    return response.choices[0].message.content
```

Test the function on a few images:

```python
for _, row in df[:5].iterrows():
    img_url = row['primary_image']
    caption = get_caption(img_url, row['title'])
    print(f"CAPTION: {caption}\n\n")
```

### Step 3: Prepare the Batch File

Create the JSONL file for batch processing.

```python
tasks = []
for index, row in df.iterrows():
    title = row['title']
    img_url = row['primary_image']
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 300,
            "messages": [
                { "role": "system", "content": caption_system_prompt },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": title },
                        { "type": "image_url", "image_url": { "url": img_url } }
                    ]
                }
            ]
        }
    }
    tasks.append(task)

file_name = "data/batch_tasks_furniture.jsonl"
with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')
```

### Step 4: Upload the File and Create a Batch Job

```python
batch_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
)

batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

### Step 5: Retrieve and Process Results

Once the job is complete, download and load the results.

```python
batch_job = client.batches.retrieve(batch_job.id)
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content

result_file_name = "data/batch_job_results_furniture.jsonl"
with open(result_file_name, 'wb') as file:
    file.write(result)

results = []
with open(result_file_name, 'r') as file:
    for line in file:
        json_object = json.loads(line.strip())
        results.append(json_object)
```

Match results to inputs using the `custom_id`:

```python
for res in results[:5]:
    task_id = res['custom_id']
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    item = df.iloc[int(index)]
    print(f"CAPTION: {result}\n\n")
```

## Conclusion

In this guide, you've learned how to use the OpenAI Batch API for two common tasks: categorizing text data and captioning images. The Batch API supports the same parameters and models as the Chat Completions endpoint (including `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, and `gpt-3.5-turbo`), making it ideal for high-volume, asynchronous processing at reduced costs. We recommend migrating eligible asynchronous workloads to the Batch API to optimize cost and efficiency.