# Incremental Prompt Engineering and Model Comparison with Mistral using Pixeltable

This guide demonstrates how to use Pixeltable for iterative prompt engineering and model comparison with Mistral AI models. You'll learn how to leverage persistent storage, incremental updates, and easily benchmark different prompts and models.

[Pixeltable](https://github.com/pixeltable/pixeltable) is a data infrastructure tool that provides a declarative, incremental approach for multimodal AI workflows.

**Category:** Prompt Engineering & Model Comparison

## Prerequisites

Before you begin, ensure you have the following:

- A Mistral AI API key
- Python installed on your system

## 1. Setup and Installation

First, install the required Python packages:

```bash
pip install -qU pixeltable mistralai textblob nltk
```

Now, import the necessary modules and set up your environment:

```python
import os
import getpass
import pixeltable as pxt
from pixeltable.functions.mistralai import chat_completions
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set your Mistral API key
if 'MISTRAL_API_KEY' not in os.environ:
    os.environ['MISTRAL_API_KEY'] = getpass.getpass('Mistral AI API Key:')
```

## 2. Create a Pixeltable Table and Insert Examples

Unlike in-memory Python libraries like Pandas, Pixeltable is a persistent database. This means your data remains available even after resetting your notebook kernel or starting a new Python session.

Let's create a table to store your prompts and input data:

```python
# Create a table to store prompts and results
pxt.drop_table('mistral_prompts', ignore_errors=True)
t = pxt.create_table('mistral_prompts', {
    'task': pxt.StringType(),
    'system': pxt.StringType(),
    'input_text': pxt.StringType()
})

# Insert sample data
t.insert([
    {'task': 'summarization',
     'system': 'Summarize the following text:',
     'input_text': 'Mistral AI is a French artificial intelligence (AI) research and development company that focuses on creating and applying AI technologies to various industries.'},
    {'task': 'sentiment',
     'system': 'Analyze the sentiment of this text:',
     'input_text': 'I love using Mistral for my AI projects! They provide great LLMs and it is really easy to work with.'},
    {'task': 'question_answering',
     'system': 'Answer the following question:',
     'input_text': 'What are the main benefits of using Mistral AI over other LLMs providers?'}
])
```

## 3. Run Mistral Inference Functions

Now you'll create **computed columns** to run Mistral's `chat_completions` function and store the outputs. Computed columns are a permanent part of the table and will automatically update whenever new data is added.

First, define the message structure using columns from your table:

```python
# Reference columns from the 'mistral_prompts' table to dynamically compose messages
msgs = [
    {'role': 'system', 'content': t.system},
    {'role': 'user', 'content': t.input_text}
]

# Run inference with open-mistral-nemo model
t['open_mistral_nemo'] = chat_completions(
    messages=msgs,
    model='open-mistral-nemo',
    max_tokens=300,
    top_p=0.9,
    temperature=0.7
)

# Run inference with mistral-medium model
t['mistral_medium'] = chat_completions(
    messages=msgs,
    model='mistral-medium',
    max_tokens=300,
    top_p=0.9,
    temperature=0.7
)
```

The response columns have JSON type. Let's extract the relevant content as strings:

```python
# Extract the response content as strings
t['omn_response'] = t.open_mistral_nemo.choices[0].message.content.astype(pxt.StringType())
t['ml_response'] = t.mistral_medium.choices[0].message.content.astype(pxt.StringType())
```

You can now view the responses:

```python
# Display the responses
t.select(t.omn_response, t.ml_response).collect()
```

## 4. Leveraging User-Defined Functions (UDFs) for Further Analysis

User-Defined Functions (UDFs) allow you to extend Pixeltable with custom Python code, enabling you to integrate any computation or analysis into your workflow.

Define three UDFs to compute metrics that provide insights into the quality of the LLM outputs:

```python
@pxt.udf
def get_sentiment_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity

@pxt.udf
def extract_keywords(text: str, num_keywords: int = 5) -> list:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    return sorted(set(keywords), key=keywords.count, reverse=True)[:num_keywords]

@pxt.udf
def calculate_readability(text: str) -> float:
    words = len(re.findall(r'\w+', text))
    sentences = len(re.findall(r'\w+[.!?]', text)) or 1
    average_words_per_sentence = words / sentences
    return 206.835 - 1.015 * average_words_per_sentence
```

Now, add these metrics as new computed columns for each model:

```python
# Add metrics for mistral-medium model
t['large_sentiment_score'] = get_sentiment_score(t.ml_response)
t['large_keywords'] = extract_keywords(t.ml_response)
t['large_readability_score'] = calculate_readability(t.ml_response)

# Add metrics for open-mistral-nemo model
t['open_sentiment_score'] = get_sentiment_score(t.omn_response)
t['open_keywords'] = extract_keywords(t.omn_response)
t['open_readability_score'] = calculate_readability(t.omn_response)
```

Once a UDF is defined and used in a computed column, Pixeltable automatically applies it to all relevant rows—no manual loops required.

## 5. Experiment with Different Prompts

Let's insert additional rows to test different prompts. Pixeltable will automatically populate all computed columns:

```python
t.insert([
    {
        'task': 'summarization',
        'system': 'Provide a concise summary of the following text in one sentence:',
        'input_text': 'Mistral AI is a company that develops AI models and has been in the news for its partnerships and latest models.'
    },
    {
        'task': 'translation',
        'system': 'Translate the following English text to French:',
        'input_text': 'Hello, how are you today?'
    }
])
```

You can filter and select specific data using `where()`:

```python
# Select specific columns for summarization tasks
t.select(
    t.task, 
    t.omn_response, 
    t.ml_response, 
    t.large_readability_score, 
    t.open_readability_score
).where(t.task == 'summarization').collect()
```

Pixeltable's schema provides a holistic view of your entire workflow, from data ingestion and inference API calls to metric computation.