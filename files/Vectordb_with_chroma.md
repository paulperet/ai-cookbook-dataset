

# Gemini API: Document Q&A with ChromaDB

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

This tutorial demonstrates how to use the Gemini API to create a vector database and retrieve answers to questions from the database. Moreover, you will use ChromaDB, an open-source Python tool that creates embedding databases. ChromaDB allows you to:

* Store embeddings as well as their metadata
* Embed documents and queries
* Search through the database of embeddings

In this tutorial, you'll use embeddings to retrieve an answer from a database of vectors created with ChromaDB.

## Setup

First, download and install ChromaDB and the Gemini API Python library.


```
%pip install -U -q "google-genai>=1.0.0" chromadb
```

Then import the modules you'll use in this tutorial.


```
import textwrap
import chromadb
import numpy as np
import pandas as pd

from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
```

## Configure your API key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in Google AI Studio.

Get an API key

In Colab, add the key to the secrets manager under the "🔑" in the left panel. Give it the name `GEMINI_API_KEY`.

Once you have the API key, pass it to the SDK. You can do this in two ways:

* Put the key in the `GEMINI_API_KEY` environment variable (the SDK will automatically pick it up from there).
* Pass the key to `genai.Client(api_key=...)`


```
from google import genai
from google.colab import userdata


GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
```

Key Point: Next, you will choose a model. Any embedding model will work for this tutorial, but for real applications it's important to choose a specific model and stick with it. The outputs of different models are not compatible with each other.

**Note**: At this time, the Gemini API is only available in certain regions.


```
for m in client.models.list():
  if 'embedContent' in m.supported_actions:
    print(m.name)
```

    [models/embedding-001, ..., models/gemini-embedding-001]


### Data

Here is a small set of documents you will use to create an embedding database:


```
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

## Creating the embedding database with ChromaDB

You will create a custom function for performing embedding using the Gemini API. By inputting a set of documents into this custom function, you will receive vectors, or embeddings of the documents.


### API changes to Embeddings with model gemini-embedding-001

For the new embeddings model, embedding-001, there is a new task type parameter and the optional title (only valid with task_type=`RETRIEVAL_DOCUMENT`).

These new parameters apply only to the newest embeddings models.The task types are:

Task Type | Description
---       | ---
RETRIEVAL_QUERY	| Specifies the given text is a query in a search/retrieval setting.
RETRIEVAL_DOCUMENT | Specifies the given text is a document in a search/retrieval setting.
SEMANTIC_SIMILARITY	| Specifies the given text will be used for Semantic Textual Similarity (STS).
CLASSIFICATION	| Specifies that the embeddings will be used for classification.
CLUSTERING	| Specifies that the embeddings will be used for clustering.


```
from google.genai import types

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    EMBEDDING_MODEL_ID = "gemini-embedding-001"  # @param ["gemini-embedding-001", "text-embedding-004"] {"allow-input": true, "isTemplate": true}
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

Now you will create the vector database. In the `create_chroma_db` function, you will instantiate a Chroma client. From there, you will create a collection, which is where you store your embeddings, documents, and any metadata. Note that the embedding function from above is passed as an argument to the `create_collection`.

Next, you use the `add` method to add the documents to the collection.


```
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
```


```
# Set up the DB
db = create_chroma_db(documents, "google-car-db")
```

    /tmp/ipykernel_83887/781126645.py:5: DeprecationWarning: The class GeminiEmbeddingFunction does not implement __init__. This will be required in a future version.
      embedding_function=GeminiEmbeddingFunction()


Confirm that the data was inserted by looking at the database:


```
sample_data = db.get(include=['documents', 'embeddings'])

df = pd.DataFrame({
    "IDs": sample_data['ids'][:3],
    "Documents": sample_data['documents'][:3],
    "Embeddings": [str(emb)[:50] + "..." for emb in sample_data['embeddings'][:3]]  # Truncate embeddings
})

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
      <th>IDs</th>
      <th>Documents</th>
      <th>Embeddings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>\n  Operating the Climate Control System  Your...</td>
      <td>[ 0.00971627 -0.00177013  0.00590323 ...  0.00...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>\n  Your Googlecar has a large touchscreen dis...</td>
      <td>[ 0.00388563 -0.00036349 -0.00230268 ...  0.01...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>\n  Shifting Gears Your Googlecar has an autom...</td>
      <td>[-0.00264773  0.010808   -0.00854844 ...  0.00...</td>
    </tr>
  </tbody>
</table>
</div>



## Getting the relevant document

`db` is a Chroma collection object. You can call `query` on it to perform a nearest neighbors search to find similar embeddings or documents.



```
def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage
```


```
# Perform embedding search
passage = get_relevant_passage("touch screen features", db)
Markdown(passage)
```





  Your Googlecar has a large touchscreen display that provides access to a
  variety of features, including navigation, entertainment, and climate
  control. To use the touchscreen display, simply touch the desired icon.
  For example, you can touch the "Navigation" icon to get directions to
  your destination or touch the "Music" icon to play your favorite songs.




Now that you have found the relevant passage in your set of documents, you can use it make a prompt to pass into the Gemini API.


```
def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""
    You are a helpful and informative bot that answers questions using
    text from the reference passage included below.
    Be sure to respond in a complete sentence, being comprehensive,
    including all relevant background information.
    However, you are talking to a non-technical audience, so be sure to
    break down complicated concepts and strike a friendly
    and converstional tone. If the passage is irrelevant to the answer,
    you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt
```

Pass a query to the prompt:


```
query = "How do you use the touchscreen in the Google car?"
prompt = make_prompt(query, passage)
Markdown(prompt)
```





    You are a helpful and informative bot that answers questions using
    text from the reference passage included below.
    Be sure to respond in a complete sentence, being comprehensive,
    including all relevant background information.
    However, you are talking to a non-technical audience, so be sure to
    break down complicated concepts and strike a friendly
    and converstional tone. If the passage is irrelevant to the answer,
    you may ignore it.
    QUESTION: 'How do you use the touchscreen in the Google car?'
    PASSAGE: '   Your Googlecar has a large touchscreen display that provides access to a   variety of features, including navigation, entertainment, and climate   control. To use the touchscreen display, simply touch the desired icon.   For example, you can touch the Navigation icon to get directions to   your destination or touch the Music icon to play your favorite songs. '

    ANSWER:
  



Now use the `generate_content` method to to generate a response from the model.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
answer = client.models.generate_content(
    model = MODEL_ID,
    contents = prompt
)
Markdown(answer.text)
```




Using the touchscreen in your Google car is super easy and intuitive! You simply **touch the icon** for whatever you want to do. This large display gives you access to lots of different features, like finding your way with **navigation**, enjoying your favorite **music or entertainment**, and even controlling the **climate** inside the car. For instance, if you want directions, you would just touch the "Navigation" icon, or if you're in the mood for some tunes, you'd tap the "Music" icon to play your favorite songs.



## Next steps

To learn more about how you can use the embeddings, check out the examples available. To learn how to use other services in the Gemini API, visit the Python quickstart.