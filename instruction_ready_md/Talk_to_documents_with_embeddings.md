# Guide: Document Search with Gemini Embeddings

## Overview
This guide demonstrates how to use the Gemini API to create embeddings for performing semantic document search. You will build a simple question-answering system that retrieves relevant information from a set of documents about a fictional "Google Car."

## Prerequisites
This tutorial requires a paid-tier Gemini API key due to rate limits. Ensure you have an API key ready.

## Setup

### 1. Install the Gemini Python Client
```bash
pip install -U "google-genai>=1.0.0"
```

### 2. Configure the Client
```python
from google import genai
from google.colab import userdata  # For Colab; adjust for your environment

GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY')  # Or set directly: GEMINI_API_KEY = "YOUR_KEY"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 1: Generate a Sample Embedding
First, let's understand how to create an embedding for a piece of text. Embeddings convert text into a numerical vector that captures its semantic meaning.

```python
from google.genai import types

EMBEDDING_MODEL_ID = "gemini-embedding-001"  # You can also use "text-embedding-004"

title = "The next generation of AI for developers and Google Workspace"
sample_text = """
Title: The next generation of AI for developers and Google Workspace
Full article:
Gemini API & Google AI Studio: An approachable way to explore and
prototype with generative AI applications
"""

embedding = client.models.embed_content(
    model=EMBEDDING_MODEL_ID,
    contents=sample_text,
    config=types.EmbedContentConfig(
        task_type="retrieval_document",
        title=title
    )
)

print(f"Embedding vector length: {len(embedding.embeddings[0].values)}")
```

The output is a high-dimensional vector (e.g., 3072 floats). This vector represents the semantic content of your text.

## Step 2: Build an Embeddings Database
Now, create a small database of documents about the Google Car and generate embeddings for each.

### Define Your Documents
```python
DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console. Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
}

DOCUMENT2 = {
    "title": "Touchscreen",
    "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon. For example, you can touch the 'Navigation' icon to get directions to your destination or touch the 'Music' icon to play your favorite songs."
}

DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position. Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."
}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
```

### Organize into a DataFrame
```python
import pandas as pd

df = pd.DataFrame(documents)
df.columns = ['Title', 'Text']
print(df)
```

### Generate Embeddings for Each Document
Define a helper function to create embeddings and add them to the DataFrame.

```python
def embed_fn(title, text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="retrieval_document",
            title=title
        )
    )
    return response.embeddings[0].values

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)
print(df[['Title', 'Embeddings']].head())
```

Your DataFrame now contains the original text and its corresponding embedding vector.

## Step 3: Perform Document Search
Now, let's search the database by comparing a question's embedding to the document embeddings.

### Understand Task Types
When using the `gemini-embedding-001` model, specify the task type:
- `RETRIEVAL_DOCUMENT`: For embedding document text.
- `RETRIEVAL_QUERY`: For embedding a user query (search string).

### Create a Query Embedding
```python
query = "How to shift gears in the Google car?"

query_embedding = client.models.embed_content(
    model=EMBEDDING_MODEL_ID,
    contents=query,
    config=types.EmbedContentConfig(task_type="retrieval_document")
)
```

### Find the Most Relevant Passage
Use the dot product to measure similarity between the query vector and each document vector. The dot product ranges from -1 (opposite) to 1 (identical direction), with higher values indicating greater similarity.

```python
import numpy as np

def find_best_passage(query, dataframe):
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
    query_embedding_response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=query,
        config=types.EmbedContentConfig(task_type="retrieval_document")
    )
    query_vector = query_embedding_response.embeddings[0].values

    # Compute dot products
    dot_products = np.dot(
        np.stack(dataframe['Embeddings']),  # Stack all document vectors
        query_vector
    )
    # Find the index of the highest similarity
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]['Text']  # Return the most relevant text

# Retrieve the best passage
passage = find_best_passage(query, df)
print("Most relevant passage:\n")
print(passage)
```

The output should be the text from the "Shifting Gears" document, as it's the most semantically similar to your query.

## Step 4: Build a Q&A Application
Finally, use a Gemini generation model to answer the question based on the retrieved passage.

### Create a Prompt Template
```python
import textwrap

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""
        You are a helpful and informative bot that answers questions using text
        from the reference passage included below. Be sure to respond in a
        complete sentence, being comprehensive, including all relevant
        background information.

        However, you are talking to a non-technical audience, so be sure to
        break down complicated concepts and strike a friendly and conversational
        tone. If the passage is irrelevant to the answer, you may ignore it.

        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

        ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

prompt = make_prompt(query, passage)
print(prompt)
```

### Generate an Answer
```python
MODEL_ID = "gemini-3-flash-preview"  # You can choose other models like "gemini-2.5-flash"

answer = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print("Answer:\n")
print(answer.text)
```

The model will generate a friendly, comprehensive answer based on the retrieved passage about shifting gears.

## Summary
You've successfully built a document search system using Gemini embeddings. The process involved:
1. Generating embeddings for a set of documents.
2. Comparing a query embedding to the document embeddings using the dot product.
3. Retrieving the most relevant passage.
4. Using a generative model to answer the question based on that passage.

## Next Steps
- Explore the [Embeddings quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb) for more details on task types and parameters.
- Browse the [cookbook examples](https://github.com/google-gemini/cookbook/tree/main/examples) for more advanced implementations like RAG pipelines or multi-document agents.