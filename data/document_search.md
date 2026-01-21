# Guide: Document Search with Embeddings using the Gemini API

This guide demonstrates how to use the Gemini API to create text embeddings and build a simple document search and question-answering system. You will learn to embed documents, compare them to a user query, and retrieve the most relevant information.

## Prerequisites

*   Python 3.11+
*   A Gemini API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup

### 1. Install the Required Library

First, install the Google Generative AI Python client library.

```bash
pip install -U google-genai
```

### 2. Import Libraries and Configure the Client

Import the necessary libraries and set up the Gemini client with your API key.

```python
import textwrap
import numpy as np
import pandas as pd
from google import genai
from google.genai import types

# Initialize the client with your API key
# Option 1: Set the GEMINI_API_KEY environment variable and use:
# client = genai.Client()
# Option 2: Pass the key directly:
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your key
client = genai.Client(api_key=GEMINI_API_KEY)
```

### 3. Choose an Embedding Model

Select an embedding model for your application. It's important to stick with one model, as embeddings from different models are not directly comparable.

```python
# List available embedding models
for m in client.models.list():
    if 'embedContent' in m.supported_actions:
        print(m.name)

# Choose your model
MODEL_ID = "gemini-embedding-001"  # Example model
```

## Step 1: Understand Embedding Task Types

The Gemini embedding model supports different task types. For document search, you will primarily use:
*   `RETRIEVAL_DOCUMENT`: For embedding the documents you want to search.
*   `RETRIEVAL_QUERY`: For embedding the user's search question.

## Step 2: Build an Embeddings Database

You will create a small database of documents related to a fictional "Google Car" and generate their embeddings.

### 2.1 Define Your Documents

Create a list of documents. Each document is a dictionary with a title and content.

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

### 2.2 Organize Documents into a DataFrame

Place the documents into a pandas DataFrame for easier management and visualization.

```python
df = pd.DataFrame(documents)
df.columns = ['Title', 'Text']
print(df)
```

### 2.3 Generate Document Embeddings

Create a helper function to embed text using the `RETRIEVAL_DOCUMENT` task type, then apply it to each document in your DataFrame.

```python
def embed_fn(text):
    """Generates embeddings for a given text string."""
    response = client.models.embed_content(
        model=MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    # The embedding values are a list of floats
    return response.embeddings[0].values

# Add an 'Embeddings' column to the DataFrame
df['Embeddings'] = df['Text'].apply(embed_fn)
print(df[['Title', 'Embeddings']].head())
```

Your database is now ready. The `Embeddings` column contains vector representations of each document's text.

## Step 3: Perform a Document Search

Now, you will ask a question and find the most relevant document by comparing the question's embedding to the document embeddings.

### 3.1 Embed the User Query

First, embed the user's question using the `RETRIEVAL_QUERY` task type.

```python
query = "How do you shift gears in the Google car?"

query_embedding_response = client.models.embed_content(
    model=MODEL_ID,
    contents=query,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)
query_embedding = query_embedding_response.embeddings[0].values
```

### 3.2 Find the Most Relevant Passage

To find the best match, you calculate the **dot product** between the query vector and each document vector. The dot product measures similarity: values closer to 1 indicate higher similarity.

```python
def find_best_passage(query, dataframe):
    """
    Finds the most relevant document for a query by comparing embeddings.
    """
    # Generate the query embedding
    query_embedding_response = client.models.embed_content(
        model=MODEL_ID,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_embedding = query_embedding_response.embeddings[0].values

    # Calculate dot products between the query and all document embeddings
    # np.stack converts the list of embedding lists into a 2D array
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding)

    # Find the index of the highest dot product
    best_index = np.argmax(dot_products)

    # Return the text of the best-matching document
    return dataframe.iloc[best_index]['Text']

# Use the function
best_passage = find_best_passage(query, df)
print("Most Relevant Passage:\n", best_passage)
```

This function will return the text from the "Shifting Gears" document, as it is the most relevant to the query about shifting gears.

## Step 4: Create a Question-Answering Application

Finally, use a Gemini text generation model to formulate a friendly, comprehensive answer based on the retrieved passage.

### 4.1 Construct a Prompt

Create a function that builds a prompt instructing the model to answer the question using the provided passage.

```python
def make_prompt(query, relevant_passage):
    """Creates a prompt for the LLM to answer a question based on a passage."""
    # Clean the passage for inclusion in the prompt
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""\
    You are a helpful and informative bot that answers questions using text from the reference passage included below.
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
    strike a friendly and conversational tone.
    If the passage is irrelevant to the answer, you may ignore it.

    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

# Generate the prompt
prompt = make_prompt(query, best_passage)
print("Prompt sent to model:\n", prompt)
```

### 4.2 Generate the Answer

Send the constructed prompt to a Gemini text generation model (like `gemini-2.5-flash`) to get the final answer.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",  # Use a suitable text generation model
    contents=prompt
)

print("Answer:\n", response.text)
```

The model will generate an answer such as: *"Shifting gears in the Google car is quite straightforward because it has an automatic transmission! All you need to do is simply move the shift lever to the position you want to be in..."*

## Summary

You have successfully built a basic Retrieval-Augmented Generation (RAG) pipeline:
1.  **Prepared a document database** and generated embeddings for each document.
2.  **Embedded a user query** and used the dot product to find the most semantically similar document.
3.  **Synthesized an answer** by providing the retrieved context to a text generation model.

This pattern forms the foundation for more advanced applications like chatbots over private documents, enhanced search systems, and intelligent agents.

## Next Steps

*   Explore other [Gemini API guides](../quickstarts/Get_started.ipynb) to learn about different models and capabilities.
*   Scale the system by storing embeddings in a dedicated vector database (e.g., Pinecone, Weaviate) for faster similarity search over larger datasets.
*   Experiment with different prompt engineering techniques to improve answer quality.