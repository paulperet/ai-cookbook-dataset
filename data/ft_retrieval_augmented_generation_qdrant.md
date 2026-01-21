# Fine-Tuning OpenAI Models for Retrieval Augmented Generation (RAG) with Qdrant and Few-Shot Learning

This guide provides a comprehensive, step-by-step tutorial on fine-tuning OpenAI models for Retrieval Augmented Generation (RAG). You will integrate Qdrant for retrieval and apply few-shot learning techniques to enhance model performance and reduce hallucinations.

**Note:** This tutorial uses the `gpt-3.5-turbo` model. While fine-tuning on the SQuAD dataset yields only minimal gains for more advanced models like GPT-4o, this workflow serves as a practical guide for implementing fine-tuning and RAG practices.

## What You Will Learn
By the end of this tutorial, you will know how to:
- Fine-tune OpenAI models for specific use cases.
- Use Qdrant to improve the retrieval component of your RAG pipeline.
- Apply fine-tuning to enhance answer correctness and reduce model hallucinations.

## Prerequisites & Setup

### 1. Install Required Libraries
First, install the necessary Python packages.

```bash
pip install pandas openai tqdm tenacity scikit-learn tiktoken python-dotenv seaborn --upgrade --quiet
```

### 2. Import Dependencies and Initialize the OpenAI Client
Import the required libraries and set up your OpenAI client. Ensure your API key is available in your environment variables.

```python
import json
import os
import time
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from openai import OpenAI
import tiktoken
from tenacity import retry, wait_exponential
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
tqdm.pandas()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### 3. Set Your Qdrant Credentials
If using Qdrant, set your cluster URL and API key as environment variables.

```python
os.environ["QDRANT_URL"] = "https://xxx.cloud.qdrant.io:6333"
os.environ["QDRANT_API_KEY"] = "xxx"
```

---

## Section A: Zero-Shot Learning Baseline

### Step 1: Prepare the SQuADv2 Dataset
You will use a subset of the SQuADv2 dataset, which includes questions where the answer may not be present in the provided context. This helps evaluate how the model handles such cases.

First, define helper functions to load and sample the data.

```python
def json_to_dataframe_with_titles(json_data):
    """Convert SQuAD JSON data to a pandas DataFrame."""
    qas = []
    context = []
    is_impossible = []
    answers = []
    titles = []

    for article in json_data['data']:
        title = article['title']
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                qas.append(qa['question'].strip())
                context.append(paragraph['context'])
                is_impossible.append(qa['is_impossible'])
                
                ans_list = []
                for ans in qa['answers']:
                    ans_list.append(ans['text'])
                answers.append(ans_list)
                titles.append(title)

    df = pd.DataFrame({
        'title': titles,
        'question': qas,
        'context': context,
        'is_impossible': is_impossible,
        'answers': answers
    })
    return df

def get_diverse_sample(df, sample_size=100, random_state=42):
    """
    Get a diverse sample by sampling from each title and impossibility status.
    """
    sample_df = df.groupby(['title', 'is_impossible']).apply(
        lambda x: x.sample(min(len(x), max(1, sample_size // 50)), random_state=random_state)
    ).reset_index(drop=True)
    
    if len(sample_df) < sample_size:
        remaining_sample_size = sample_size - len(sample_df)
        remaining_df = df.drop(sample_df.index).sample(remaining_sample_size, random_state=random_state)
        sample_df = pd.concat([sample_df, remaining_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sample_df.sample(min(sample_size, len(sample_df)), random_state=random_state).reset_index(drop=True)
```

Load the validation dataset and create a sample of 100 diverse examples.

```python
# Load the validation data (assumes files are downloaded to 'local_cache/')
val_df = json_to_dataframe_with_titles(json.load(open('local_cache/dev.json')))
df = get_diverse_sample(val_df, sample_size=100, random_state=42)
```

### Step 2: Define the Zero-Shot Prompt
Create a function that formats the prompt for the base model. The instruction is to answer based solely on the provided context.

```python
def get_prompt(row):
    """Generate the zero-shot prompt for a given row."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
Question: {row.question}\n\n
Context: {row.context}\n\n
Answer:\n""",
        },
    ]
```

### Step 3: Create a Robust API Call Function
Define a function to call the OpenAI API with retry logic using `tenacity`.

```python
@retry(wait=wait_exponential(multiplier=1, min=2, max=6))
def api_call(messages, model):
    """Make an API call to OpenAI with exponential backoff retries."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["\n\n"],
        max_tokens=100,
        temperature=0.0,
    )

def answer_question(row, prompt_func=get_prompt, model="gpt-3.5-turbo"):
    """Generate an answer for a given row using the specified prompt and model."""
    messages = prompt_func(row)
    response = api_call(messages, model)
    return response.choices[0].message.content
```

### Step 4: Generate Answers with the Base Model
Now, apply the `answer_question` function to your validation sample to get baseline predictions.

```python
# Generate answers using the base model
df["generated_answer"] = df.progress_apply(answer_question, axis=1)

# Save the results
df.to_json("local_cache/100_val.json", orient="records", lines=True)
df = pd.read_json("local_cache/100_val.json", orient="records", lines=True)
```

### Step 5: Prepare Data for Fine-Tuning
To fine-tune the model, you need a training dataset in the required JSONL format. Use a sample from the training split.

```python
def dataframe_to_jsonl(df):
    """Convert a DataFrame to the JSONL format required for OpenAI fine-tuning."""
    def create_jsonl_entry(row):
        answer = row["answers"][0] if row["answers"] else "I don't know"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
Question: {row.question}\n\n
Context: {row.context}\n\n
Answer:\n""",
            },
            {"role": "assistant", "content": answer},
        ]
        return json.dumps({"messages": messages})

    jsonl_output = df.apply(create_jsonl_entry, axis=1)
    return "\n".join(jsonl_output)

# Load and sample the training data
train_df = json_to_dataframe_with_titles(json.load(open('local_cache/train.json')))
train_sample = get_diverse_sample(train_df, sample_size=100, random_state=42)

# Save the training data as a JSONL file
with open("local_cache/100_train.jsonl", "w") as f:
    f.write(dataframe_to_jsonl(train_sample))
```

### Step 6: Fine-Tune the OpenAI Model
Create a class to manage the fine-tuning process, including file upload and job creation.

```python
class OpenAIFineTuner:
    """A class to handle the fine-tuning process for OpenAI models."""
    def __init__(self, training_file_path, model_name, suffix):
        self.training_file_path = training_file_path
        self.model_name = model_name
        self.suffix = suffix
        self.file_object = None
        self.fine_tuning_job = None
        self.model_id = None

    def create_openai_file(self):
        """Upload the training file to OpenAI."""
        self.file_object = client.files.create(
            file=open(self.training_file_path, "rb"),
            purpose="fine-tune",
        )

    def wait_for_file_processing(self, sleep_time=20):
        """Wait until the uploaded file is processed by OpenAI."""
        while self.file_object.status != 'processed':
            time.sleep(sleep_time)
            self.file_object.refresh()
            print("File Status: ", self.file_object.status)

    def create_fine_tuning_job(self):
        """Create a fine-tuning job using the uploaded file."""
        self.fine_tuning_job = client.fine_tuning.jobs.create(
            training_file=self.file_object.id,
            model=self.model_name,
            suffix=self.suffix
        )
```

To execute the fine-tuning:
1. Instantiate the `OpenAIFineTuner` class with your training file, base model (`gpt-3.5-turbo`), and a unique suffix.
2. Upload the file and wait for processing.
3. Create the fine-tuning job.

Refer to the [OpenAI Fine-Tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning) for detailed instructions on monitoring job status and using the fine-tuned model.

---

## Next Steps: Section B - Few-Shot Learning with Qdrant

In the next section, you will:
1. Integrate Qdrant to retrieve relevant context snippets dynamically.
2. Enhance the prompt with few-shot examples retrieved from Qdrant.
3. Fine-tune the model using this improved, retrieval-augmented pipeline.
4. Evaluate the performance gains compared to the zero-shot baseline.

This approach combines the strengths of dense retrieval (Qdrant) with the generative power of a fine-tuned LLM, creating a robust RAG system capable of handling complex queries with higher accuracy and fewer hallucinations.