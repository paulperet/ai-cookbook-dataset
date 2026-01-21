# Guide: Adding Context to Gemini API Queries

## Overview
Large Language Models (LLMs) like Gemini are trained on vast datasets, but they cannot know everything. When you need to query information that is new, private, or not widely available, you must provide that context directly in your prompt. This guide demonstrates how to structure a prompt with custom context to get accurate answers from the Gemini API.

## Prerequisites

### 1. Install the Gemini Python Client
First, install the required library.

```bash
pip install -U "google-genai>=1.0.0"
```

### 2. Import Required Modules
```python
from google import genai
from IPython.display import Markdown
```

### 3. Configure Your API Key
Store your Gemini API key in an environment variable or a secure secret manager. This example assumes you have stored it in a Colab Secret named `GOOGLE_API_KEY`.

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Select a Model
Choose a Gemini model for your task. You can modify the `MODEL_ID` variable as needed.

```python
MODEL_ID = "gemini-3-flash-preview"  # Example model
```

## Step-by-Step Tutorial: Querying with Custom Context

### Step 1: Define Your Context and Query
Construct a prompt that clearly separates your query from the context data. This helps the model understand which information is provided for reference.

In this example, we use a small table of Olympic athlete participation counts.

```python
prompt = """
QUERY: Provide a list of athletes that competed in the Olympics exactly 9 times.
CONTEXT:

Table title: Olympic athletes and number of times they've competed
Ian Millar, 10
Hubert Raudaschl, 9
Afanasijs Kuzmins, 9
Nino Salukvadze, 9
Piero d'Inzeo, 8
Raimondo d'Inzeo, 8
Claudia Pechstein, 8
Jaqueline Mourão, 8
Ivan Osiier, 7
François Lafortune, Jr, 7
"""
```

**Why this structure works:** The `QUERY:` and `CONTEXT:` labels create a clear boundary. The model uses the table data (which it has never seen before) to answer the specific question.

### Step 2: Call the Gemini API
Pass your structured prompt to the model using the `generate_content` method.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)
```

### Step 3: Display the Result
The model will parse the context table and return only the names matching your criteria.

```python
Markdown(response.text)
```

**Expected Output:**
The model should return a list like:
* Hubert Raudaschl
* Afanasijs Kuzmins
* Nino Salukvadze

It correctly ignores athletes with 8, 10, or 7 appearances, demonstrating precise use of the provided context.

## Practical Applications

This technique is essential for working with:
* **Private Documentation:** Internal company wikis, process guides, or proprietary manuals.
* **Financial Records:** Data from QuickBooks, spreadsheets, or internal reports.
* **Specialized Forums:** Archived discussions or community knowledge bases not indexed by search engines.
* **Real-time Data:** Statistics, logs, or sensor data generated after the model's training cutoff.

## Next Steps & Further Exploration

1.  **Classify Your Own Data:** Use this template to feed the model examples of product reviews, support tickets, or inventory items and ask it to categorize them.
2.  **Experiment with Few-Shot Prompting:** Extend this concept by providing several example question-answer pairs in your context before asking a new, similar question.
3.  **Explore Advanced Prompting:** Check the [Gemini Cookbook repository](https://github.com/google-gemini/cookbook) for more examples on chain-of-thought reasoning, structured output generation, and multi-turn conversations.

By mastering the skill of injecting context, you enable the Gemini API to act as a powerful reasoning engine over your specific, private datasets.