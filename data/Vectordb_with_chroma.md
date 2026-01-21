# Gemini API Cookbook: Document Q&A with ChromaDB

## Overview
This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) system using the Gemini API and ChromaDB. You will learn to create a vector database from documents and retrieve contextually relevant answers to user queries.

## Prerequisites
Ensure you have a Gemini API key. If you don't have one, you can create it in [Google AI Studio](https://aistudio.google.com/app/apikey). This tutorial requires paid tier rate limits to run properly.

## Setup
Install the required Python libraries.

```bash
pip install -U "google-genai>=1.0.0" chromadb pandas numpy
```

Import the necessary modules.

```python
import textwrap
import chromadb
import numpy as np
import pandas as pd
from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
```

## Step 1: Configure the Gemini API Client
Set up your API key and initialize the Gemini client. If you're using Google Colab, you can store your key in the secrets manager under the name `GEMINI_API_KEY`.

```python
# If using in a local environment, set your API key as an environment variable
# import os
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# For Colab, use:
from google.colab import userdata
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 2: Choose an Embedding Model
List the available embedding models from the Gemini API. It's important to choose a specific model and stick with it, as outputs from different models are not compatible.

```python
for m in client.models.list():
    if 'embedContent' in m.supported_actions:
        print(m.name)
```

## Step 3: Prepare Your Documents
Define the sample documents you'll use to populate your vector database. These documents contain information about a fictional "Googlecar."

```python
DOCUMENT1 = """
  Operating the Climate Control System  Your Googlecar has a climate control
  system that allows you to adjust the temperature and airflow in the car.
  To operate the climate control system, use the buttons and knobs located on
  the center console.  Temperature: The temperature knob controls the
  temperature inside the car. Turn the knob clockwise to increase the
  temperature or counterclockwise to decrease the temperature.
  Airflow: The airflow knob controls the amount of airflow inside the car.
  Turn the knob clockwise to increase the airflow or counterclockwise to
  decrease the airflow. Fan speed: The fan speed knob controls the speed
  of the fan. Turn the knob clockwise to increase the fan speed or
  counterclockwise to decrease the fan speed.
  Mode: The mode button allows you to select the desired mode. The available
  modes are: Auto: The car will automatically adjust the temperature and
  airflow to maintain a comfortable level.
  Cool: The car will blow cool air into the car.
  Heat: The car will blow warm air into the car.
  Defrost: The car will blow warm air onto the windshield to defrost it.
"""

DOCUMENT2 = """
  Your Googlecar has a large touchscreen display that provides access to a
  variety of features, including navigation, entertainment, and climate
  control. To use the touchscreen display, simply touch the desired icon.
  For example, you can touch the \"Navigation\" icon to get directions to
  your destination or touch the \"Music\" icon to play your favorite songs.
"""

DOCUMENT3 = """
  Shifting Gears Your Googlecar has an automatic transmission. To
  shift gears, simply move the shift lever to the desired position.
  Park: This position is used when you are parked. The wheels are locked
  and the car cannot move.
  Reverse: This position is used to back up.
  Neutral: This position is used when you are stopped at a light or in traffic.
  The car is not in gear and will not move unless you press the gas pedal.
  Drive: This position is used to drive forward.
  Low: This position is used for driving in snow or other slippery conditions.
"""

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
```

## Step 4: Create a Custom Embedding Function
Define a custom `EmbeddingFunction` class for ChromaDB that uses the Gemini API to generate embeddings. This function will convert your documents into vector representations.

**Note:** The `gemini-embedding-001` model introduces a new `task_type` parameter. For document retrieval, use `RETRIEVAL_DOCUMENT`.

```python
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        EMBEDDING_MODEL_ID = "gemini-embedding-001"
        title = "Custom query"
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title=title
            )
        )
        return response.embeddings[0].values
```

## Step 5: Build the Vector Database with ChromaDB
Create a function to instantiate a ChromaDB client, create a collection, and add your documents. The collection stores the documents, their embeddings, and associated metadata.

```python
def create_chroma_db(documents, name):
    chroma_client = chromadb.Client()
    db = chroma_client.create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction()
    )

    for i, d in enumerate(documents):
        db.add(
            documents=d,
            ids=str(i)
        )
    return db

# Initialize the database
db = create_chroma_db(documents, "google-car-db")
```

## Step 6: Verify the Database Contents
Confirm that your documents and their embeddings have been successfully stored.

```python
sample_data = db.get(include=['documents', 'embeddings'])

df = pd.DataFrame({
    "IDs": sample_data['ids'][:3],
    "Documents": sample_data['documents'][:3],
    "Embeddings": [str(emb)[:50] + "..." for emb in sample_data['embeddings'][:3]]
})

print(df)
```

## Step 7: Retrieve Relevant Documents
Define a function to query the database. This performs a nearest-neighbor search to find the document most similar to your query.

```python
def get_relevant_passage(query, db):
    passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
    return passage

# Test the retrieval
query = "touch screen features"
passage = get_relevant_passage(query, db)
print(passage)
```

## Step 8: Construct a Prompt for the LLM
Create a function that formats the retrieved passage and the user's query into a prompt suitable for the Gemini model. This prompt instructs the model to answer conversationally using the provided context.

```python
def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""
        You are a helpful and informative bot that answers questions using
        text from the reference passage included below.
        Be sure to respond in a complete sentence, being comprehensive,
        including all relevant background information.
        However, you are talking to a non-technical audience, so be sure to
        break down complicated concepts and strike a friendly
        and conversational tone. If the passage is irrelevant to the answer,
        you may ignore it.
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

        ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

# Generate the prompt
query = "How do you use the touchscreen in the Google car?"
prompt = make_prompt(query, passage)
print(prompt)
```

## Step 9: Generate the Final Answer
Pass the constructed prompt to a Gemini generative model to produce a final, user-friendly answer.

```python
MODEL_ID = "gemini-3-flash-preview"  # You can choose other available models
answer = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(answer.text)
```

## Next Steps
You have successfully built a basic RAG pipeline. To expand this project:
* Experiment with larger document sets and chunking strategies.
* Integrate metadata filtering in your ChromaDB queries.
* Explore other embedding models and compare their performance.
* For more examples and advanced use cases, refer to the [Gemini API Python quickstart](https://ai.google.dev/gemini-api/docs/quickstart?lang=python).