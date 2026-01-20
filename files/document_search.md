# Document search with embeddings

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

This example demonstrates how to use the Gemini API to create embeddings so that you can perform document search. You will use the Python client library to build a word embedding that allows you to compare search strings, or questions, to document contents.

In this tutorial, you'll use embeddings to perform document search over a set of documents to ask questions related to the Google Car.

## Prerequisites

You can run this quickstart in Google Colab.

To complete this quickstart on your own development environment, ensure that your environment meets the following requirements:

-  Python 3.11+
-  An installation of `jupyter` to run the notebook.

## Setup

First, download and install the Gemini API Python library.


```
!pip install -U -q google-genai
```


```
import textwrap
import numpy as np
import pandas as pd

from google import genai
from google.genai import types

# Used to securely store your API key
from google.colab import userdata

from IPython.display import Markdown
```

### Grab an API Key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in Google AI Studio.

Get an API key

In Colab, add the key to the secrets manager under the "🔑" in the left panel. Give it the name `GEMINI_API_KEY`.

Once you have the API key, pass it to the SDK. You can do this in two ways:

* Put the key in the `GEMINI_API_KEY` environment variable (the SDK will automatically pick it up from there).
* Pass the key to `genai.Client(api_key=...)`


```
# Or use `os.getenv('GEMINI_API_KEY')` to fetch an environment variable.
GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
```

Key Point: Next, you will choose a model. Any embedding model will work for this tutorial, but for real applications it's important to choose a specific model and stick with it. The outputs of different models are not compatible with each other.


```
for m in client.models.list():
  if 'embedContent' in m.supported_actions:
    print(m.name)
```

    [models/embedding-001, ..., models/gemini-embedding-001]


### Select the model to be used


```
MODEL_ID = "gemini-embedding-001" # @param ["gemini-embedding-001", "text-embedding-004"] {"allow-input":true, isTemplate: true}
```

## Generate the embeddings

In this section, you will see how to generate embeddings for the different texts in the dataframe using the embeddings from the Gemini API.

The Gemini embedding model supports several task types, each tailored for a specific goal. Here’s a general overview of the available types and their applications:

Task Type | Description
---       | ---
RETRIEVAL_QUERY	| Specifies the given text is a query in a search/retrieval setting.
RETRIEVAL_DOCUMENT | Specifies the given text is a document in a search/retrieval setting.
SEMANTIC_SIMILARITY	| Specifies the given text will be used for Semantic Textual Similarity (STS).
CLASSIFICATION	| Specifies that the embeddings will be used for classification.
CLUSTERING	| Specifies that the embeddings will be used for clustering.


```
sample_text = ("Title: The next generation of AI for developers and Google Workspace"
    "\n"
    "Full article:\n"
    "\n"
    "Gemini API & Google AI Studio: An approachable way to explore and prototype with generative AI applications")

embedding = client.models.embed_content(model=MODEL_ID,
                                contents=sample_text,
                                config=types.EmbedContentConfig(
                                  task_type="RETRIEVAL_DOCUMENT"))

print(embedding.embeddings)
```

    [ContentEmbedding(
      values=[
        -0.009020552,
        0.0153440945,
        0.0027249781,
        -0.07818188,
        0.003901859,
        <... 3067 more items ...>,
      ]
    )]


## Building an embeddings database

Here are three sample texts to use to build the embeddings database. You will use the Gemini API to create embeddings of each of the documents. Turn them into a dataframe for better visualization.


```
DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."}
DOCUMENT2 = {
    "title": "Touchscreen",
    "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."}
DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
```

Organize the contents of the dictionary into a dataframe for better visualization.


```
df = pd.DataFrame(documents)
df.columns = ['Title', 'Text']
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
      <th>Title</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Operating the Climate Control System</td>
      <td>Your Googlecar has a climate control system th...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Touchscreen</td>
      <td>Your Googlecar has a large touchscreen display...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shifting Gears</td>
      <td>Your Googlecar has an automatic transmission. ...</td>
    </tr>
  </tbody>
</table>
</div>



Get the embeddings for each of these bodies of text. Add this information to the dataframe.


```
# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(text):
  return client.models.embed_content(model=MODEL_ID,
                             contents=text,
                             config=types.EmbedContentConfig(
                               task_type="RETRIEVAL_DOCUMENT")
                             )

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Text']).embeddings[0].values, axis=1)
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
      <th>Title</th>
      <th>Text</th>
      <th>Embeddings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Operating the Climate Control System</td>
      <td>Your Googlecar has a climate control system th...</td>
      <td>[0.027014425, -0.0028718826, 0.015998857, -0.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Touchscreen</td>
      <td>Your Googlecar has a large touchscreen display...</td>
      <td>[0.018501397, -0.004494585, 0.0063248016, -0.0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shifting Gears</td>
      <td>Your Googlecar has an automatic transmission. ...</td>
      <td>[0.010804788, 0.020962104, -0.0016377118, -0.0...</td>
    </tr>
  </tbody>
</table>
</div>



## Document search with Q&A

Now that the embeddings are generated, let's create a Q&A system to search these documents. You will ask a question about hyperparameter tuning, create an embedding of the question, and compare it against the collection of embeddings in the dataframe.

The embedding of the question will be a vector (list of float values), which will be compared against the vector of the documents using the dot product. This vector returned from the API is already normalized. The dot product represents the similarity in direction between two vectors.

The values of the dot product can range between -1 and 1, inclusive. If the dot product between two vectors is 1, then the vectors are in the same direction. If the dot product value is 0, then these vectors are orthogonal, or unrelated, to each other. Lastly, if the dot product is -1, then the vectors point in the opposite direction and are not similar to each other.

Note, with the new embeddings model (`embedding-001`), specify the task type as `QUERY` for user query and `DOCUMENT` when embedding a document text.

Task Type | Description
---       | ---
RETRIEVAL_QUERY	| Specifies the given text is a query in a search/retrieval setting.
RETRIEVAL_DOCUMENT | Specifies the given text is a document in a search/retrieval setting.


```
query = "How do you shift gears in the Google car?"
model = MODEL_ID

request = client.models.embed_content(model=model,
                                contents=query,
                                config=types.EmbedContentConfig(
                                  task_type="RETRIEVAL_QUERY"))
```

Use the `find_best_passage` function to calculate the dot products, and then sort the dataframe from the largest to smallest dot product value to retrieve the relevant passage out of the database.


```
def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = client.models.embed_content(model=model,
                                                contents=query,
                                                config=types.EmbedContentConfig(
                                                  task_type="RETRIEVAL_QUERY")).embeddings[0].values
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding)
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Text'] # Return text from index with max value
```

View the most relevant document from the database:


```
passage = find_best_passage(query, df)
passage
```




    'Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions.'



## Question and Answering Application

Let's try to use the text generation API to create a Q & A system. Input your own custom data below to create a simple question and answering example. You will still use the dot product as a metric of similarity.


```
def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt
```


```
prompt = make_prompt(query, passage)
print(prompt)
```

    You are a helpful and informative bot that answers questions using text from the reference passage included below.   Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.   However, you are talking to a non-technical audience, so be sure to break down complicated concepts and   strike a friendly and converstional tone.   If the passage is irrelevant to the answer, you may ignore it.
      QUESTION: 'How do you shift gears in the Google car?'
      PASSAGE: 'Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions.'
    
        ANSWER:
    


Choose one of the Gemini content generation models in order to find the answer to your query.


```
answer = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(answer.text)
```

    Shifting gears in the Google car is quite straightforward because it has an automatic transmission! All you need to do is simply move the shift lever to the position you want to be in. For example, if you're parked, you'd use the "Park" position, which locks the wheels to keep the car from moving. When you want to back up, you'll choose "Reverse." If you're stopped at a light or stuck in traffic and want the car to stay still without being in gear, you'd select "Neutral"; in this mode, the car won't move unless you press the gas pedal. To drive forward, you'll simply put it in "Drive." And if you ever find yourself driving in challenging conditions like snow or on slippery roads, there's a "Low" position that can help with that!



```
Markdown(answer.text)
```




Shifting gears in the Google car is quite straightforward because it has an automatic transmission! All you need to do is simply move the shift lever to the position you want to be in. For example, if you're parked, you'd use the "Park" position, which locks the wheels to keep the car from moving. When you want to back up, you'll choose "Reverse." If you're stopped at a light or stuck in traffic and want the car to stay still without being in gear, you'd select "Neutral"; in this mode, the car won't move unless you press the gas pedal. To drive forward, you'll simply put it in "Drive." And if you ever find yourself driving in challenging conditions like snow or on slippery roads, there's a "Low" position that can help with that!



## Next steps

To learn how to use other services in the Gemini API, see the [Get started](../quickstarts/Get_started.ipynb) guide.