# Data Generation: Refining News Articles

In this cookbook, we will dig into the process of generating data to fine-tune a model for rewriting articles in a specific, refined format. We will utilize a two-step pipeline for this purpose. First, we will generate critiques about the articles making use of guides that our model should respect and use as reference. Then, using these critiques, we will produce new, refined articles. The goal is to create a dataset that includes at least the original article and its refined version, which could potentially be used to fine-tune a model in the future or other purposes.

You can download the guides for this notebook with the following:

```python
!wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_1.txt
!wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_2.txt
!wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_3.txt
!wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_4.txt
```

First step is to install `mistralai` and create a client with your api key!

```python
!pip install mistralai==0.4.2
```

```python
from mistralai.client import MistralClient

# Other imports we will need
from tqdm.contrib.concurrent import process_map
import secrets
import time
import random
import json
import os
```

```python
CLIENT = MistralClient(api_key="api_key")
```

The next step is to download the dataset. We will be making use of a dataset available on Hugging Face, but you could provide your own!

```python
!pip install datasets
```

For this example, we will be generating 100 pairs of the original articles and the refined ones, but you are free to generate as many as you require.

```python
import datasets
news_articles = list(datasets.load_dataset("AyoubChLin/CNN_News_Articles_2011-2022", split="train"))

random.shuffle(news_articles)

print("Total Articles:", len(news_articles))

n_sample = 100
news_articles = random.sample(news_articles, n_sample)

print("Sampled:", n_sample)

with open("./news.jsonl", "w") as f:
  for news in news_articles:
    f.write(json.dumps({"news": news["text"]}) + "\n")
```

Our pipeline will consist of two steps. First, we will generate critiques using a style guideline of our choice. Here, we have four different guidelines that are more or less the same, but you could rewrite your own.

Once the critiques have been generated, we will use them to generate the new rewritten articles!

Let's get started with the criticism!

Let's create a folder where we will cache our data as we generate it. This can be handy for debugging and to have a backup in case something goes wrong.

```python
import os
newpath = r'./data'
if not os.path.exists(newpath):
    os.makedirs(newpath)
```

Now, let's define the first process. We will make use of `mistral-large-latest` capabilities to both criticize and rewrite our articles, but you are free to use any combination of your choice.

```python
def process_critique(args):
    line, systems, guides = args
    record = json.loads(line)

    news_article = record.get("news")

    part = random.choice(list(range(len(guides))))
    guide = guides[part]

    part = random.choice(list(range(len(systems))))
    system = systems[part].format(guide)

    time.sleep(1)
    try:
        answer = CLIENT.chat(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": news_article},
            ],
            temperature=0.2,
            max_tokens=2048
        )
        critique = answer.choices[0].message.content

        result = json.dumps({"news": news_article, "critique": critique, "status": "SUCCESS"})

    except Exception as e:
        result = json.dumps({"news": news_article, "critique": str(e), "status": "ERROR"})

    random_hash = secrets.token_hex(4)

    with open(f"./data/news_critique_{random_hash}.jsonl", "w") as f:
        f.write(result)

    return result
```

To generate diverse output each time, it might be a good idea to have multiple system prompts instead of a single one. Here, we provide a few system prompts that are all very similar but overall different.

```python
systems_variations = [
    "As a 'News Article Editor' adhering to a specific style guide, your responsibility is to polish and restructure news articles to align them with the high standards of clarity, accuracy, and elegance set by the guide:\n\n {} \n\n You are presented with a news article. Identify the ten (or fewer) most significant stylistic concerns and provide examples of how they can be enhanced.",
    "As a 'News Content Refiner' committed to the guide, your role is to revise and perfect news articles to ensure they meet the exceptional standards of lucidity, exactness, and refinement synonymous with the guide:\n\n {} \n\n You have a news article at hand. Pinpoint the sixteen (or less) most crucial stylistic problems and suggest examples of how they might be improved.",
    "As a 'News Piece Stylist' in accordance with the style guide, your duty is to amend and enrich news articles to guarantee they adhere to the rigorous standards of clarity, precision, and sophistication embodied by the style guide:\n\n {} \n\n You are handed a news piece. Highlight the fourteen (or fewer) most pressing stylistic errors and offer examples of how they could be rectified.",
    "As a 'News Article Enhancer' following the principles of the guide, your mission is to modify and elevate news articles to match the high-quality standards of clarity, precision, and eloquence established by the style guide:\n\n {} \n\n You are given a news article to work on. Specify the twenty (or less) most notable stylistic flaws and provide examples of how they can be bettered.",
    "As a 'News Prose Stylist' abiding by the style guide, your assignment is to correct and embellish news articles to ensure they meet the distinguished standards of clarity, precision, and sophistication upheld by the guide:\n\n {} \n\n You are provided with a news article for evaluation. Indicate the twenty (or fewer) most important stylistic issues and propose examples of how they may be optimized.",
    "As a 'News Report Stylist' in compliance with the guide, your job is to revise and improve news articles to guarantee they align with the high benchmarks of clarity, precision, and sophistication set forth by the guide:\n\n {} \n\n You are tasked with reviewing a news report. List the fifteen (or less) most critical stylistic shortcomings and provide examples of how they might be amended.",
    "As a 'News Writing Stylist' in line with the guide, your responsibility is to edit and refine news articles to ensure they meet the superior standards of clarity, precision, and sophistication inherent to the style guide:\n\n {} \n\n You are assigned to edit a news article. Identify the sixteen (or fewer) most prominent stylistic inconsistencies and suggest examples of how they can be enhanced.",
    "As a 'News Text Stylist' adhering to the style guide, your role is to amend and perfect news articles to ensure they meet the high-caliber standards of clarity, precision, and sophistication characteristic of the guide:\n\n {} \n\n You are given a news text to evaluate. Highlight the nineteen (or less) most significant stylistic discrepancies and provide examples of how they might be improved.",
    "As a 'News Copy Stylist' in accordance with the guide, your duty is to revise and enrich news articles to guarantee they adhere to the exacting standards of clarity, precision, and sophistication embodied by the style guide:\n\n {} \n\n You are tasked with reviewing a news copy. List the eleven (or fewer) most crucial stylistic errors and propose examples of how they can be rectified.",
    "As a 'News Article Stylist and Editor' committed to the style guide, your mission is to refine, rewrite, and edit news articles to ensure they meet the high standards of clarity, precision, and sophistication synonymous with the guide:\n\n {} \n\n You are given a news article to refine and edit. Identify the seventeen (or fewer) most pressing stylistic concerns and provide examples of how they can be improved."
]
```

Now, it's time to generate. Let's get the guides we made and start the generation using `process_map`, which will create multiple workers to generate the new data in parallel and more efficiently.

```python
guides = []
for pick in range(1, 5):
    jsonl_file_path = f"./guide_{pick}.txt"

    with open(jsonl_file_path, "r") as f:
        guide = f.read()
        guides.append(guide)

data_path = "./news.jsonl"
with open(data_path, "r") as f:
    lines = f.readlines()
    lines = [(line, systems_variations, guides) for line in lines]

    results = process_map(process_critique, lines, max_workers=20, chunksize=1)

with open("./generated_news_critiques.jsonl", "w") as f:
    for result in results:
        f.write(result + "\n")
```

Perfect! Critiques generated, now it's time to refine and rewrite our articles using the feedback!

```python
def process_refined_news(args):
    line, system, instruction = args
    record = json.loads(line)

    news_article = record.get("news")
    critique= record.get("critique")
    status = record.get("status")

    time.sleep(1)

    try:
      if status == "SUCCESS":

        answer = CLIENT.chat(
            model="mistral-large-latest",
            messages= [
                {"role": "system", "content": system},
                {"role": "user", "content": news_article},
                {"role": "assistant", "content": critique},
                {"role": "user", "content": instruction},
            ],
            temperature=0.2,
            max_tokens=2048
        )
        new_news = answer.choices[0].message.content

        result = json.dumps({"news": news_article, "critique": critique, "refined_news": new_news, "status": "SUCCESS"})

      else:
        result = json.dumps({"news": news_article, "critique": critique, "refined_news": critique, "status": "ERROR"})
    except Exception as e:
        result = json.dumps({"news": news_article, "critique": critique, "refined_news": str(e), "status": "ERROR"})

    random_hash = secrets.token_hex(4)

    with open(f"./data/refined_news_{random_hash}.jsonl", "w") as f:
        f.write(result)

    return result
```

We will replace our multiple system variations with a generalized one to give it context, but the key part of our second step is our instruction to rewrite the article with the provided feedback. This instruction might require a lot of changes depending on your requirements!

```python
system = "Polish and restructure the news articles to align them with the high standards of clarity, accuracy, and elegance set by the style guide. You are presented with a news article. Identify the ten (or fewer) most significant stylistic concerns and provide examples of how they can be enhanced."

instruction = """
Now, I want you to incorporate the feedback and critiques into the news article and respond with the enhanced version, focusing solely on stylistic improvements without altering the content.
You must provide the entire article enhanced.
Do not make ANY comments, only provide the new article improved.
Do not tell me what you changed, only provide the new article taking into consideration the feedback you provided.
The new article needs to have all the content of the original article but with the feedback into account.
"""

data_path = "./generated_news_critiques.jsonl"
with open(data_path, "r") as f:
    lines = f.readlines()
    lines = [(line, system, instruction) for line in lines]

    results = process_map(process_refined_news, lines, max_workers=20, chunksize=1)

with open("./generated_refined_news.jsonl", "w") as f:
    for result in results:
        f.write(result + "\n")
```

Articles generated! Let's take a look at them.

```python
from pprint import pprint

with open("./generated_refined_news.jsonl", "r") as f:
  l = json.loads(f.readlines()[12])

pprint(l)
```