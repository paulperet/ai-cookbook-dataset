# Fine-Tuning OpenAI Models for Retrieval Augmented Generation (RAG) with Qdrant and Few-Shot Learning

The aim of this notebook is to walk through a comprehensive example of how to fine-tune OpenAI models for Retrieval Augmented Generation (RAG). 

We will also be integrating Qdrant and Few-Shot Learning to boost the model's performance and reduce hallucinations. This could serve as a practical guide for ML practitioners, data scientists, and AI Engineers interested in leveraging the power of OpenAI models for specific use-cases. 🤩

Note: This notebook uses the gpt-3.5-turbo model. Fine-tuning on the SQuAD dataset with this setup yields only minimal gains for more advanced models such as gpt-4o or gpt-4.1. As such, this notebook is primarily intended as a guide for fine-tuning workflows and retrieval-augmented generation (RAG) practices

## Why should you read this blog?

You want to learn how to 
- [Fine-tune OpenAI models](https://platform.openai.com/docs/guides/fine-tuning/) for specific use-cases
- Use [Qdrant](https://qdrant.tech/documentation/) to improve the performance of your RAG model
- Use fine-tuning to improve the correctness of your RAG model and reduce hallucinations

To begin, we've selected a dataset where we've a guarantee that the retrieval is perfect. We've selected a subset of the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset, which is a collection of questions and answers about Wikipedia articles. We've also included samples where the answer is not present in the context, to demonstrate how RAG handles this case.

## Table of Contents
1. Setting up the Environment

### Section A: Zero-Shot Learning
2. Data Preparation: SQuADv2 Dataset
3. Answering using Base gpt-3.5-turbo-0613 model
4. Fine-tuning and Answering using Fine-tuned model
5. **Evaluation**: How well does the model perform?

### Section B: Few-Shot Learning

6. Using Qdrant to Improve RAG Prompt
7. Fine-Tuning OpenAI Model with Qdrant
8. Evaluation

9. **Conclusion**
    - Aggregate Results
    - Observations

## Terms, Definitions, and References

**Retrieval Augmented Generation (RAG)?**
The phrase Retrieval Augmented Generation (RAG) comes from a [recent paper](https://arxiv.org/abs/2005.11401) by Lewis et al. from Facebook AI. The idea is to use a pre-trained language model (LM) to generate text, but to use a separate retrieval system to find relevant documents to condition the LM on. 

**What is Qdrant?**
Qdrant is an open-source vector search engine that allows you to search for similar vectors in a large dataset. It is built in Rust and here we'll use the Python client to interact with it. This is the Retrieval part of RAG.

**What is Few-Shot Learning?**
Few-shot learning is a type of machine learning where the model is "improved" via training or fine-tuning on a small amount of data. In this case, we'll use it to fine-tune the RAG model on a small number of examples from the SQuAD dataset. This is the Augmented part of RAG.

**What is Zero-Shot Learning?**
Zero-shot learning is a type of machine learning where the model is "improved" via training or fine-tuning without any dataset specific information. 

**What is Fine-Tuning?**
Fine-tuning is a type of machine learning where the model is "improved" via training or fine-tuning on a small amount of data. In this case, we'll use it to fine-tune the RAG model on a small number of examples from the SQuAD dataset. The LLM is what makes the Generation part of RAG.

## 1. Setting Up the Environment

### Install and Import Dependencies


```python
!pip install pandas openai tqdm tenacity scikit-learn tiktoken python-dotenv seaborn --upgrade --quiet
```


```python
import json
import os
import time

import pandas as pd
from openai import OpenAI
import tiktoken
import seaborn as sns
from tenacity import retry, wait_exponential
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

```

### Set your keys
Get your OpenAI keys [here](https://platform.openai.com/account/api-keys) and Qdrant keys after making a free cluster [here](https://cloud.qdrant.io/login).


```python
os.environ["QDRANT_URL"] = "https://xxx.cloud.qdrant.io:6333"
os.environ["QDRANT_API_KEY"] = "xxx"
```

## Section A

## 2. Data Preparation: SQuADv2 Data Subsets

For the purpose of demonstration, we'll make small slices from the train and validation splits of the [SQuADv2](https://rajpurkar.github.io/SQuAD-explorer/) dataset. This dataset has questions and contexts where the answer is not present in the context, to help us evaluate how LLM handles this case.

We'll read the data from the JSON files and create a dataframe with the following columns: `question`, `context`, `answer`, `is_impossible`.

### Download the Data


```python
# !mkdir -p local_cache
# !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O local_cache/train.json
# !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O local_cache/dev.json
```

### Read JSON to DataFrame


```python
def json_to_dataframe_with_titles(json_data):
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

    df = pd.DataFrame({'title': titles, 'question': qas, 'context': context, 'is_impossible': is_impossible, 'answers': answers})
    return df

def get_diverse_sample(df, sample_size=100, random_state=42):
    """
    Get a diverse sample of the dataframe by sampling from each title
    """
    sample_df = df.groupby(['title', 'is_impossible']).apply(lambda x: x.sample(min(len(x), max(1, sample_size // 50)), random_state=random_state)).reset_index(drop=True)
    
    if len(sample_df) < sample_size:
        remaining_sample_size = sample_size - len(sample_df)
        remaining_df = df.drop(sample_df.index).sample(remaining_sample_size, random_state=random_state)
        sample_df = pd.concat([sample_df, remaining_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sample_df.sample(min(sample_size, len(sample_df)), random_state=random_state).reset_index(drop=True)

train_df = json_to_dataframe_with_titles(json.load(open('local_cache/train.json')))
val_df = json_to_dataframe_with_titles(json.load(open('local_cache/dev.json')))

df = get_diverse_sample(val_df, sample_size=100, random_state=42)
```

## 3. Answering using Base gpt-3.5-turbo-0613 model

### 3.1 Zero Shot Prompt

Let's start by using the base gpt-3.5-turbo-0613 model to answer the questions. This prompt is a simple concatenation of the question and context, with a separator token in between: `\n\n`. We've a simple instruction part of the prompt: 

> Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.

Other prompts are possible, but this is a good starting point. We'll use this prompt to answer the questions in the validation set. 


```python
# Function to get prompt messages
def get_prompt(row):
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

### 3.2 Answering using Zero Shot Prompt

Next, you'll need some re-usable functions which make an OpenAI API Call and return the answer. You'll use the `ChatCompletion.create` endpoint of the API, which takes a prompt and returns the completed text.


```python
# Function with tenacity for retries
@retry(wait=wait_exponential(multiplier=1, min=2, max=6))
def api_call(messages, model):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["\n\n"],
        max_tokens=100,
        temperature=0.0,
    )


# Main function to answer question
def answer_question(row, prompt_func=get_prompt, model="gpt-3.5-turbo"):
    messages = prompt_func(row)
    response = api_call(messages, model)
    return response.choices[0].message.content
```

⏰ **Time to run: ~3 min**, 🛜 Needs Internet Connection


```python
# Use progress_apply with tqdm for progress bar
df["generated_answer"] = df.progress_apply(answer_question, axis=1)
df.to_json("local_cache/100_val.json", orient="records", lines=True)
df = pd.read_json("local_cache/100_val.json", orient="records", lines=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>question</th>
      <th>context</th>
      <th>is_impossible</th>
      <th>answers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Scottish_Parliament</td>
      <td>What consequence of establishing the Scottish ...</td>
      <td>A procedural consequence of the establishment ...</td>
      <td>False</td>
      <td>[able to vote on domestic legislation that app...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Imperialism</td>
      <td>Imperialism is less often associated with whic...</td>
      <td>The principles of imperialism are often genera...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Economic_inequality</td>
      <td>What issues can't prevent women from working o...</td>
      <td>When a person’s capabilities are lowered, they...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Southern_California</td>
      <td>What county are Los Angeles, Orange, San Diego...</td>
      <td>Its counties of Los Angeles, Orange, San Diego...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>French_and_Indian_War</td>
      <td>When was the deportation of Canadians?</td>
      <td>Britain gained control of French Canada and Ac...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Geology</td>
      <td>In the layered Earth model, what is the inner ...</td>
      <td>Seismologists can use the arrival times of sei...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Prime_number</td>
      <td>What type of value would the Basel function ha...</td>
      <td>The zeta function is closely related to prime ...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Fresno,_California</td>
      <td>What does the San Joaquin Valley Railroad cros...</td>
      <td>Passenger rail service is provided by Amtrak S...</td>
      <td>True</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Victoria_(Australia)</td>
      <td>What party rules in Melbourne's inner regions?</td>
      <td>The centre-left Australian Labor Party (ALP), ...</td>
      <td>False</td>
      <td>[The Greens, Australian Greens, Greens]</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Immune_system</td>
      <td>The speed of the killing response of the human...</td>
      <td>In humans, this response is activated by compl...</td>
      <td>False</td>
      <td>[signal amplification, signal amplification, s...</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



## 4. Fine-tuning and Answering using Fine-tuned model

For the complete fine-tuning process, please refer to the [OpenAI Fine-Tuning Docs](https://platform.openai.com/docs/guides/fine-tuning/use-a-fine-tuned-model).

### 4.1 Prepare the Fine-Tuning Data

We need to prepare the data for fine-tuning. We'll use a few samples from train split of same dataset as before, but we'll add the answer to the context. This will help the model learn to retrieve the answer from the context. 

Our instruction prompt is the same as before, and so is the system prompt. 


```python
def dataframe_to_jsonl(df):
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

train_sample = get_diverse_sample(train_df, sample_size=100, random_state=42)

with open("local_cache/100_train.jsonl", "w") as f:
    f.write(dataframe_to_jsonl(train_sample))
```

**Tip: 💡 Verify the Fine-Tuning Data**

You can see this [cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Chat_finetuning_data_prep.ipynb) for more details on how to prepare the data for fine-tuning.

### 4.2 Fine-Tune OpenAI Model

If you're new to OpenAI Model Fine-Tuning, please refer to the [How to finetune Chat models](https://github.com/openai/openai-cookbook/blob/448a0595b84ced3bebc9a1568b625e748f9c1d60/examples/How_to_finetune_chat_models.ipynb) notebook. You can also refer to the [OpenAI Fine-Tuning Docs](platform.openai.com/docs/guides/fine-tuning/use-a-fine-tuned-model) for more details.


```python
class OpenAIFineTuner:
    """
    Class to fine tune OpenAI models
    """
    def __init__(self, training_file_path, model_name, suffix):
        self.training_file_path = training_file_path
        self.model_name = model_name
        self.suffix = suffix
        self.file_object = None
        self.fine_tuning_job = None
        self.model_id = None

    def create_openai_file(self):
        self.file_object = client.files.create(
            file=open(self.training_file_path, "rb"),
            purpose="fine-tune",
        )

    def wait_for_file_processing(self, sleep_time=20):
        while self.file_object.status != 'processed':
            time.sleep(sleep_time)
            self.file_object.refresh()
            print("File Status: ", self.file_object.status)

    def create_fine_tuning_job(self):
        self.fine_tuning_job = client.fine_tuning.jobs.create(
            training_file=self.file_object.id,
            model=self.model_name,
            suffix=self.s