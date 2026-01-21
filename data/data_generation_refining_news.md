# Guide: Refining News Articles with a Two-Step AI Pipeline

This guide walks you through creating a dataset for fine-tuning a model to rewrite news articles in a refined style. You will implement a two-step pipeline: first, generating critiques based on specific style guides, then using those critiques to produce polished articles. The result is a dataset of original and refined article pairs.

## Prerequisites

Ensure you have the necessary libraries installed and your API key ready.

```bash
pip install mistralai==0.4.2 datasets
```

```python
from mistralai.client import MistralClient
from tqdm.contrib.concurrent import process_map
import secrets
import time
import random
import json
import os

# Initialize the Mistral client with your API key
CLIENT = MistralClient(api_key="your_api_key_here")
```

## Step 1: Download Style Guides

Download the four style guides that will inform the critique and refinement process.

```bash
wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_1.txt
wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_2.txt
wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_3.txt
wget https://github.com/mistralai/cookbook/blob/main/mistral/data_generation/external_files/guide_4.txt
```

## Step 2: Load and Sample the News Dataset

You will use a dataset of CNN news articles from Hugging Face. For this example, you'll sample 100 articles, but you can adjust this number.

```python
import datasets

# Load the dataset
news_articles = list(datasets.load_dataset("AyoubChLin/CNN_News_Articles_2011-2022", split="train"))
random.shuffle(news_articles)

print(f"Total Articles: {len(news_articles)}")

# Sample 100 articles
n_sample = 100
news_articles = random.sample(news_articles, n_sample)
print(f"Sampled: {n_sample}")

# Save the sampled articles to a JSONL file
with open("./news.jsonl", "w") as f:
    for news in news_articles:
        f.write(json.dumps({"news": news["text"]}) + "\n")
```

## Step 3: Set Up a Data Directory

Create a directory to cache intermediate results, which is useful for debugging and backup.

```python
newpath = r'./data'
if not os.path.exists(newpath):
    os.makedirs(newpath)
```

## Step 4: Define the Critique Generation Function

This function takes an article, a random system prompt, and a random style guide, then uses the Mistral model to generate a critique.

```python
def process_critique(args):
    line, systems, guides = args
    record = json.loads(line)
    news_article = record.get("news")

    # Randomly select a guide and a system prompt variation
    guide = random.choice(guides)
    system = random.choice(systems).format(guide)

    time.sleep(1)  # Rate limiting
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

    # Save the result with a random hash
    random_hash = secrets.token_hex(4)
    with open(f"./data/news_critique_{random_hash}.jsonl", "w") as f:
        f.write(result)

    return result
```

## Step 5: Define System Prompt Variations

To ensure diverse critiques, define multiple system prompts. Each prompts the model to act as an editor following the provided style guide.

```python
systems_variations = [
    "As a 'News Article Editor' adhering to a specific style guide, your responsibility is to polish and restructure news articles to align them with the high standards of clarity, accuracy, and elegance set by the guide:\n\n {} \n\n You are presented with a news article. Identify the ten (or fewer) most significant stylistic concerns and provide examples of how they can be enhanced.",
    "As a 'News Content Refiner' committed to the guide, your role is to revise and perfect news articles to ensure they meet the exceptional standards of lucidity, exactness, and refinement synonymous with the guide:\n\n {} \n\n You have a news article at hand. Pinpoint the sixteen (or less) most crucial stylistic problems and suggest examples of how they might be improved.",
    # ... (additional variations follow the same pattern)
    "As a 'News Article Stylist and Editor' committed to the style guide, your mission is to refine, rewrite, and edit news articles to ensure they meet the high standards of clarity, precision, and sophistication synonymous with the guide:\n\n {} \n\n You are given a news article to refine and edit. Identify the seventeen (or fewer) most pressing stylistic concerns and provide examples of how they can be improved."
]
```

## Step 6: Generate Critiques

Load the style guides and run the critique generation in parallel using `process_map` for efficiency.

```python
# Load the four style guides
guides = []
for pick in range(1, 5):
    with open(f"./guide_{pick}.txt", "r") as f:
        guides.append(f.read())

# Load the sampled articles
data_path = "./news.jsonl"
with open(data_path, "r") as f:
    lines = f.readlines()
    # Prepare arguments for parallel processing
    lines = [(line, systems_variations, guides) for line in lines]

    # Generate critiques using 20 parallel workers
    results = process_map(process_critique, lines, max_workers=20, chunksize=1)

# Save all critiques to a single file
with open("./generated_news_critiques.jsonl", "w") as f:
    for result in results:
        f.write(result + "\n")
```

## Step 7: Define the Article Refinement Function

This function takes the original article and its generated critique, then produces a refined version of the article.

```python
def process_refined_news(args):
    line, system, instruction = args
    record = json.loads(line)

    news_article = record.get("news")
    critique = record.get("critique")
    status = record.get("status")

    time.sleep(1)  # Rate limiting

    try:
        if status == "SUCCESS":
            answer = CLIENT.chat(
                model="mistral-large-latest",
                messages=[
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
            # If critique generation failed, propagate the error
            result = json.dumps({"news": news_article, "critique": critique, "refined_news": critique, "status": "ERROR"})
    except Exception as e:
        result = json.dumps({"news": news_article, "critique": critique, "refined_news": str(e), "status": "ERROR"})

    # Save the result with a random hash
    random_hash = secrets.token_hex(4)
    with open(f"./data/refined_news_{random_hash}.jsonl", "w") as f:
        f.write(result)

    return result
```

## Step 8: Refine the Articles

Define the system prompt and instruction for the refinement step, then process all critiques to generate the final refined articles.

```python
# System prompt for the refinement step
system = "Polish and restructure the news articles to align them with the high standards of clarity, accuracy, and elegance set by the style guide. You are presented with a news article. Identify the ten (or fewer) most significant stylistic concerns and provide examples of how they can be enhanced."

# Instruction for incorporating the critique
instruction = """
Now, I want you to incorporate the feedback and critiques into the news article and respond with the enhanced version, focusing solely on stylistic improvements without altering the content.
You must provide the entire article enhanced.
Do not make ANY comments, only provide the new article improved.
Do not tell me what you changed, only provide the new article taking into consideration the feedback you provided.
The new article needs to have all the content of the original article but with the feedback into account.
"""

# Load the generated critiques
data_path = "./generated_news_critiques.jsonl"
with open(data_path, "r") as f:
    lines = f.readlines()
    # Prepare arguments for parallel processing
    lines = [(line, system, instruction) for line in lines]

    # Generate refined articles using 20 parallel workers
    results = process_map(process_refined_news, lines, max_workers=20, chunksize=1)

# Save all refined articles to a single file
with open("./generated_refined_news.jsonl", "w") as f:
    for result in results:
        f.write(result + "\n")
```

## Step 9: Inspect the Results

Finally, examine a sample entry from your generated dataset to verify the output.

```python
from pprint import pprint

with open("./generated_refined_news.jsonl", "r") as f:
    # Load the 13th entry (index 12) as an example
    sample_entry = json.loads(f.readlines()[12])
    pprint(sample_entry)
```

## Conclusion

You have successfully created a pipeline that generates critiques of news articles based on style guides and uses those critiques to produce refined versions. The final dataset, saved in `generated_refined_news.jsonl`, contains pairs of original and refined articles, ready for potential use in fine-tuning a language model or other applications.