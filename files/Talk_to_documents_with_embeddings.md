

# Document search with embeddings

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

This example demonstrates how to use the Gemini API to create embeddings so that you can perform document search. You will use the Python client library to build a word embedding that allows you to compare search strings, or questions, to document contents.

In this tutorial, you'll use embeddings to perform document search over a set of documents to ask questions related to the Google Car.

## Setup

```
%pip install -U -q "google-genai>=1.0.0"
```

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.

```
from google import  genai
from google.colab import userdata

GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Embedding generation

In this section, you will see how to generate embeddings for a piece of text using the embeddings from the Gemini API.

See the [Embeddings quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb) to learn more about the `task_type` parameter used below.

```
from google.genai import types

title = "The next generation of AI for developers and Google Workspace"
sample_text = """
    Title: The next generation of AI for developers and Google Workspace
    Full article:
    Gemini API & Google AI Studio: An approachable way to explore and
    prototype with generative AI applications
"""

EMBEDDING_MODEL_ID = MODEL_ID = "gemini-embedding-001"  # @param ["gemini-embedding-001", "text-embedding-004"] {"allow-input": true, "isTemplate": true}
embedding = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=sample_text,
        config=types.EmbedContentConfig(
            task_type="retrieval_document",
            title=title
    ))

print(embedding)
```

    embeddings=[ContentEmbedding(
      values=[
        -0.019380787,
        0.015025399,
        0.006310311,
        -0.057478663,
        0.011998727,
        <... 3067 more items ...>,
      ]
    )] metadata=None

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
import pandas as pd

df = pd.DataFrame(documents)
df.columns = ['Title', 'Text']
df
```




Title	Text
0	Operating the Climate Control System	Your Googlecar has a climate control system th...
1	Touchscreen	Your Googlecar has a large touchscreen display...
2	Shifting Gears	Your Googlecar has an automatic transmission. ...

Get the embeddings for each of these bodies of text. Add this information to the dataframe.

```
# Get the embeddings of each text and add to an embeddings column in the dataframe
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
df
```




Title	Text	Embeddings
0	Operating the Climate Control System	Your Googlecar has a climate control system th...	[0.02483931, -0.003871694, 0.013593362, -0.031...
1	Touchscreen	Your Googlecar has a large touchscreen display...	[0.008149438, -0.0013574613, -0.0029458047, -0...
2	Shifting Gears	Your Googlecar has an automatic transmission. ...	[0.009464946, 0.022619268, -0.0036155856, -0.0...

## Document search with Q&A

Now that the embeddings are generated, let's create a Q&A system to search these documents. You will ask a question about hyperparameter tuning, create an embedding of the question, and compare it against the collection of embeddings in the dataframe.

The embedding of the question will be a vector (list of float values), which will be compared against the vector of the documents using the dot product. This vector returned from the API is already normalized. The dot product represents the similarity in direction between two vectors.

The values of the dot product can range between -1 and 1, inclusive. If the dot product between two vectors is 1, then the vectors are in the same direction. If the dot product value is 0, then these vectors are orthogonal, or unrelated, to each other. Lastly, if the dot product is -1, then the vectors point in the opposite direction and are not similar to each other.

Note, with the new embeddings model (`gemini-embedding-001`), specify the task type as `QUERY` for user query and `DOCUMENT` when embedding a document text.

Task Type | Description
---       | ---
RETRIEVAL_QUERY	| Specifies the given text is a query in a search/retrieval setting.
RETRIEVAL_DOCUMENT | Specifies the given text is a document in a search/retrieval setting.

```
query = "How to shift gears in the Google car?"

request = client.models.embed_content(
    model=EMBEDDING_MODEL_ID,
    contents=query,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        )
)
```

Use the `find_best_passage` function to calculate the dot products, and then sort the dataframe from the largest to smallest dot product value to retrieve the relevant passage out of the database.

```
import numpy as np

def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = client.models.embed_content(
      model=EMBEDDING_MODEL_ID,
      contents=query,
      config=types.EmbedContentConfig(
          task_type="retrieval_document",
          )
  )

  dot_products = np.dot(
      np.stack(dataframe['Embeddings']),
      query_embedding.embeddings[0].values
  )
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Text'] # Return text from index with max value
```

View the most relevant document from the database:

```
from IPython.display import Markdown

passage = find_best_passage(query, df)
Markdown(passage)
```




Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions.

## Question and Answering Application

Let's try to use the text generation API to create a Q & A system. Input your own custom data below to create a simple question and answering example. You will still use the dot product as a metric of similarity.

```
import textwrap

def make_prompt(query, relevant_passage):
  escaped = (
      relevant_passage
      .replace("'", "")
      .replace('"', "")
      .replace("\n", " ")
  )
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
```

```
prompt = make_prompt(query, passage)
Markdown(prompt)
```





You are a helpful and informative bot that answers questions using text
from the reference passage included below. Be sure to respond in a
complete sentence, being comprehensive, including all relevant
background information.

However, you are talking to a non-technical audience, so be sure to
break down complicated concepts and strike a friendly and conversational
tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: 'How to shift gears in the Google car?'
PASSAGE: 'Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions.'

ANSWER:




Choose one of the Gemini content generation models in order to find the answer to your query.

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
answer = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)
```

```
Markdown(answer.text)
```




Good news! Your Googlecar actually has an automatic transmission, which means shifting gears is super simple – you just move the shift lever to the spot you need! For instance, if you're parked and want the car to stay put, you'll put it in 'Park' because that locks the wheels. When you need to back up, you'll choose 'Reverse.' If you're stopped at a traffic light or in slow traffic and don't want the car to roll, 'Neutral' is the spot; the car won't move unless you press the gas pedal. To drive forward, you'll simply select 'Drive.' And for those times when you're driving in snow or really slippery conditions, there's a 'Low' position to help you out.

## Next steps

Check out the [embeddings quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb) to learn more, and browse the cookbook for more [examples](https://github.com/google-gemini/cookbook/tree/main/examples).