# Incremental Prompt Engineering and Model Comparison with Mistral using Pixeltable

This notebook shows how to use Pixeltable for iterative prompt engineering and model comparison with Mistral AI models. It showcases persistent storage, incremental updates, and how to benchmark different prompts and models easily.

[Pixeltable](https://github.com/pixeltable/pixeltable) is data infrastructure that provides a declarative, incremental approach for multimodal AI.

**Category:** Prompt Engineering & Model Comparison

## 1. Setup and Installation

```python
%pip install -qU pixeltable mistralai textblob nltk
```

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
```

```python
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

```python
if 'MISTRAL_API_KEY' not in os.environ:
    os.environ['MISTRAL_API_KEY'] = getpass.getpass('Mistral AI API Key:')
```

## 2. Create a Pixeltable Table and Insert Examples

First, Pixeltable is persistent. Unlike in-memory Python libraries such as Pandas, Pixeltable is a database. When you reset a notebook kernel or start a new Python session, you'll have access to all the data you've stored previously in Pixeltable.

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

We create **computed columns** to instruct Pixeltable to run the Mistral `chat_completions` function and store the output. Because computed columns are a permanent part of the table, they will be automatically updated any time new data is added to the table. For more information, see our [tutorial](https://docs.pixeltable.com/docs/computed-columns).

In this particular example we are running the `open_mistral_nemo` and `mistral_medium` models and make the output available in their respective columns.

```python
# We are referencing columns from the 'mistral_prompts' table to dynamically compose the message for the Inference API.
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

The respective response columns have the JSON column type and we can now use JSON path expressions to extract the relevant pieces of data and make them available as additional computed columns.

```python
# Extract the response content as a string (by default JSON)
t['omn_response'] = t.open_mistral_nemo.choices[0].message.content.astype(pxt.StringType())
t['ml_response'] = t.mistral_medium.choices[0].message.content.astype(pxt.StringType())
```

```python
# Display the responses
t.select(t.omn_response, t.ml_response).collect()
```

We can see how data is computed across the different columns in our table.

```python
t
```

## 4. Leveraging User-Defined Functions (UDFs) for Further Analysis

UDFs allow you to extend Pixeltable with custom Python code, enabling you to integrate any computation or analysis into your workflow. See our [tutorial](https://docs.pixeltable.com/docs/user-defined-functions-udfs) regarding UDFs to learn more.

We define three UDFs to compute two metrics (sentiment and readability scores) that give us insights into the quality of the LLM outputs.

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

For each model we want to compare we are adding the metrics as new computed columns, using the UDFs we created.

```python
t['large_sentiment_score'] = get_sentiment_score(t.ml_response)
t['large_keywords'] = extract_keywords(t.ml_response)
t['large_readability_score'] = calculate_readability(t.ml_response)

t['open_sentiment_score'] = get_sentiment_score(t.omn_response)
t['open_keywords'] = extract_keywords(t.omn_response)
t['open_readability_score'] = calculate_readability(t.omn_response)
```

 Once a UDF is defined and used in a computed column, Pixeltable automatically applies it to all relevant rows.

 You don't need to write loops or worry about applying the function to each row manually.

```python
t.head(1)
```

## 5. Experiment with Different Prompts

We are inserting an additional two rows, and Pixeltable will automatically populate the computed columns.

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

Often you want to select only certain rows and/or certain columns in a table. You can do this with `where()`.

You can learn more about the available table and data operations [here](https://docs.pixeltable.com/docs/tables-and-data-operations).

```python
t.select(t.task, t.omn_response, t.ml_response, t.large_readability_score, t.open_readability_score).where(t.task == 'summarization').collect()
```

Pixeltable's schema provides a holistic view of data ingestion, inference API calls, and metric computation, reflecting your entire workflow.

```python
t
```