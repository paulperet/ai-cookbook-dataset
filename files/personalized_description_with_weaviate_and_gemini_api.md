##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Personalized Product Descriptions with Weaviate and the Gemini API

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

Weaviate is an open-source vector database that enables you to build AI-powered applications with the Gemini API! This notebook has four parts:
1. [Part 1: Connect to Weaviate, Define Schema, and Import Data](#part-1-install-dependencies-and-connect-to-weaviate)

2. [Part 2: Run Vector Search Queries](#part-2-vector-search)

3. [Part 3: Generative Feedback Loops](#part-3-generative-feedback-loops)

4. [Part 4: Personalized Product Descriptions](#part-4-personalization)


In this demo, you will learn how to embed your data, run a semantic search, make a generative call to the Gemini API and store the output in your vector database, and personalize the description based on the user profile.

## Use Case

You will be working with an e-commerce dataset containing Google merch. You will load the data into the Weaviate vector database and use the semantic search features to retrieve data. Next, you'll generate product descriptions and store them back into the database with a vector embedding for retrieval (aka, generative feedback loops). Lastly, you'll create a small knowledge graph with uniquely generated product descriptions for the buyer personas Alice and Bob.

## Requirements
You will need a running Weaviate cluster and Gemini API key. You'll set up these requirements as you progress through this notebook!

1. Weaviate vector database
    1. Serverless
    1. Embedded
    1. Local (Docker)
1. Gemini API key
    1. Create an API key via [AI Studio](https://aistudio.google.com/)

## Video
**For an awesome walk through of this demo, check out [this](https://youtu.be/WORgeRAAN-4?si=-WvqNkPn8oCmnLGQ&t=1138) presentation from Google Cloud Next!**

## Install Dependencies and Libraries


```
%pip install weaviate-client==4.7.1
%pip install -U -q "google-genai>=1.0.0"
%pip install requests
%pip install 'protobuf>=5'
```


```
import weaviate
from weaviate.classes.config import Configure
from weaviate.embedded import EmbeddedOptions
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, ReferenceProperty
from weaviate.util import generate_uuid5
from weaviate.classes.query import QueryReference

import os
import json
import requests
import PIL
import IPython

from PIL import Image
from io import BytesIO
from IPython.display import Markdown

# Convert image links to PIL object
def url_to_pil(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# display images
def display_image(url, size=100):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)

    resized_image = image.resize((size,size))

    display(resized_image)
```

## Part 1: Connect to Weaviate, Define Schema, and Import Data

### Connect to Weaviate

You will need to create a Weaviate cluster. There are a few ways to do this:

1. [Weaviate Embedded](https://weaviate.io/developers/weaviate/installation/embedded): Run Weaviate in your runtime

2. [Weaviate Cloud](console.weaviate.cloud): Create a sandbox on our managed service. You will need to deploy it in US West, US East, or Australia.

3. Local Host: [Docker](https://weaviate.io/developers/weaviate/installation/docker-compose#starter-docker-compose-file) or [Kubernetes](https://weaviate.io/developers/weaviate/installation/kubernetes)

For the full list of installation options, see [this page](https://weaviate.io/developers/weaviate/installation).

#### Weaviate Embedded
You will default to Weaviate Embedded. This runs Weaviate inside your notebook and is ideal for quick experimentation.

**Note: It will disconnect once you stop the terminal.**

**Set up your API key**

To run the following cell, your Gemini API key must be stored in a Colab Secret and named `GEMINI_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
from google import genai
from google.colab import userdata

GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
```


```
client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(
        version="1.25.10",
        additional_env_vars={
            "ENABLE_MODULES": "text2vec-palm, generative-palm"
        }),
        additional_headers={
            "X-Google-Studio-Api-Key": GEMINI_API_KEY
        }
)

client.connect()
```

    [INFO:weaviate-client:Binary /root/.cache/weaviate-embedded did not exist. Downloading binary from https://github.com/weaviate/weaviate/releases/download/v1.25.10/weaviate-v1.25.10-Linux-amd64.tar.gz, INFO:weaviate-client:Started /root/.cache/weaviate-embedded: process ID 716]

#### Other Options: Weaviate Cloud and Local Host

#### **Weaviate Cloud**

You can connect your notebook to a serverless Weaviate cluster to keep the data persistent in the cloud. You can register [here](https://console.weaviate.cloud/) and create a free 14-day sandbox!

To connect to your WCD cluster:
```python
WCD_URL = "https://sandbox.gcp.weaviate.cloud"
WCD_AUTH_KEY = "sk-key"
GEMINI_API_KEY = "sk-key"

client = weaviate.connect_to_wcs(
    cluster_url=WCD_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCD_AUTH_KEY),
    headers={"X-Google-Studio-Api-Key": GEMINI_API_KEY},
)

print(client.is_ready())
```

#### **Local Host**

If you want to run Weaviate yourself, you can download the [Docker files](https://weaviate.io/developers/weaviate/installation/docker-compose) and run it locally on your machine or in the cloud. There is also a `yaml` file in this folder you can use.

To connect to Weaviate locally:
```python
client = weaviate.connect_to_local()

print(client.is_ready())
```

### Create schema
The schema tells Weaviate how you want to store your data.

You will first create two collections: Products and Personas. Each collection has metadata (properties) and specifies the embedding and language model.

In [Part 4](#part-4-personalization), you will create another collection, `Personalized`, that will generate product descriptions based on the persona.


```
# This is optional to empty your database
result = client.collections.delete("Products")
print(result)
result = client.collections.delete("Personas")
print(result)
```

    None
    None



```
PROJECT_ID = "" # leave this empty
API_ENDPOINT = "generativelanguage.googleapis.com"
embedding_model = "text-embedding-004" # embedding model
generative_model = "gemini-3-flash-preview" # language model

# Products Collection
if not client.collections.exists("Products"):
  collection = client.collections.create(
    name="Products",
    vectorizer_config=Configure.Vectorizer.text2vec_palm
    (
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = embedding_model
    ),
    generative_config=Configure.Generative.palm(
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = generative_model
    ),
    properties=[ # properties for the Products collection
            Property(name="product_id", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="link", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="brand", data_type=DataType.TEXT),
            Property(name="generated_description", data_type=DataType.TEXT),
      ]
  )

# Personas Collection
if not client.collections.exists("Personas"):
  collection = client.collections.create(
    name="Personas",
    vectorizer_config=Configure.Vectorizer.text2vec_palm
    (
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = embedding_model
    ),
    generative_config=Configure.Generative.palm(
        project_id=PROJECT_ID,
        api_endpoint=API_ENDPOINT,
        model_id = generative_model
    ),
    properties=[ # properties for the Personas collection
            Property(name="name", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
      ]
  )
```

### Import Objects


```
# URL to the raw JSON file
url = 'https://raw.githubusercontent.com/bkauf/next-store/main/first_99_objects.json'
response = requests.get(url)

# Load the entire JSON content
data = json.loads(response.text)
```


```
# Print first object
data[0]
```




    {'id': 'id_1',
     'product_id': 'GGOEGAYC135814',
     'title': 'Google Badge Tee',
     'category': 'Apparel  Accessories Tops  Tees Tshirts',
     'link': 'https://shop.googlemerchandisestore.com/store/20160512512/assets/items/images/GGOEGXXX1358.jpg',
     'description': 'A classic crew neck tee made from 100 cotton Its soft and comfortable and features a small Google logo on the chest',
     'color': "['Blue']",
     'gender': 'Unisex',
     'brand': 'Google'}



#### Upload to Weaviate

To make sure everything is set, you will upload only one object and confirm it's in the database.


```
products = client.collections.get("Products")

first_object = data[0]

products.data.insert(
    properties={
        "product_id": first_object['product_id'],
        "title": first_object['title'],
        "category": first_object['category'],
        "link": first_object['link'],
        "description": first_object['description'],
        "brand": first_object['brand']
    }
)

response = products.aggregate.over_all(total_count=True)
print(response.total_count) # This should output 1
```

    1


Let's import the remainder of our dataset. You will use Weaviate's batch import to get the 98 objects into our database.


```
products = client.collections.get("Products")

remaining_data = data[1:]

with products.batch.dynamic() as batch:
  for item in remaining_data:
    batch.add_object(
      properties={
        "product_id": item['product_id'],
        "title": item['title'],
        "category": item['category'],
        "link": item['link'],
        "description": item['description'],
        "brand": item['brand']
    }
  )

response = products.aggregate.over_all(total_count=True)
print(response.total_count) # this should print 99
```

    99



```
# print the first object uuid and properties to analyze how data is structured
first_product = next(products.iterator())
print(first_product.uuid,"\n", json.dumps(first_product.properties, indent=4))
```

    000db09a-3e33-462d-be03-5715d9cde529 
     {
        "description": "The unisex Waze logo hoodie is a perfect way to show your love for the Waze app The hoodie is made of a soft and comfortable cotton blend and features a kangaroo pocket and a drawstring hood The Waze logo is embroidered on the front of the hoodie",
        "generated_description": null,
        "product_id": "GGCPWAEJ402614",
        "link": "https://shop.googlemerchandisestore.com/store/20190522377/assets/items/images/GGCPWXXX4026.jpg",
        "category": "Clothing Unisex Clothing Hoodies  Sweatshirts",
        "title": "Unisex Waze Logo Hoodie",
        "brand": "Waze"
    }


You will fetch the object by the UUID that was created. It will print out the vector embedding as well!


```
product = products.query.fetch_object_by_id(
    first_product.uuid,
    include_vector=True
)

print(f"Product: {product.properties['title']}")
print(f"Vector Dimensionality: {len(product.vector['default'])}")
print(f"Vector Preview: {product.vector['default'][:50]} ...")
```

    Product: Unisex Waze Logo Hoodie
    Vector Dimensionality: 768
    Vector Preview: [0.000302837259368971, -0.009837448596954346, 0.022877082228660583, -0.012845730409026146, 0.03479452058672905, 0.06852656602859497, 0.023823168128728867, -0.008967291563749313, -0.007616781629621983, 0.03446043282747269, -0.03289780020713806, -0.041297584772109985, 0.047338180243968964, -0.007135987281799316, 0.03441719338297844, -0.09767945110797882, -0.02126118168234825, 0.041591379791498184, -0.04611879959702492, -0.040249183773994446, -0.032080017030239105, -0.04292743653059006, 0.03954688459634781, -0.04586683586239815, -0.008462391793727875, 0.010463248938322067, 0.021219832822680473, 0.04991120845079422, 0.00274534709751606, -0.02430199831724167, 0.07378753274679184, 0.015373442322015762, -0.03711485117673874, -0.021959325298666954, 0.030712256208062172, -0.029576774686574936, -0.023318929597735405, 0.003632798558101058, -0.012349340133368969, -0.054480548948049545, -0.037265971302986145, 0.04853177070617676, -0.04757307469844818, -0.00198557460680604, 0.027546513825654984, 0.03101896308362484, -0.02321699447929859, 0.01798059605062008, 0.011385709047317505, 0.07581991702318192] ...


## Part 2: Vector Search

### Vector Search
Vector search returns the objects with most similar vectors to that of the query. You will use the `near_text` operator to find objects with the nearest vector to an input text.


```
products = client.collections.get("Products")

response = products.query.near_text(
        query="travel mug",
        return_properties=["title", "description", "link"], # only return these 3 properties
        limit=3 # limited to 3 objects
)

for product in response.objects:
    print(json.dumps(product.properties, indent=2))
    display_image(product.properties['link'])
    print('===')
```

    {
      "title": "Google Campus Bike Mug",
      "description": "The Google Campus Bike Corkbase Mug is a blue mug with a cork bottom It features a design of a yellow bicycle with a basket on the front The mug is perfect for coffee tea or any other hot beverage",
      "link": "https://shop.googlemerchandisestore.com/store/20160512512/assets/items/images/GGOEGDWC122799.jpg"
    }
    ===
    {
      "title": "Google San Francisco Mug",
      "description": "The Google San Francisco Mug is a black and white mug with the Google logo on it It is made of ceramic and is dishwasher safe",
      "link": "https://shop.googlemerchandisestore.com/store/20160512512/assets/items/images/GGOEGDWJ146799.jpg"
    }
    ===
    {
      "description": "The YETI Rambler 20 oz Tumbler is made of stainless steel and has a doublewall vacuum insulation to keep drinks cold for up to 24 hours and hot for up to 12 hours It is also dishwashersafe for easy cleaning The tumbler is silver and has the Google Cloud Certified logo on it",
      "title": "Google Cloud Certified Professional Security Engineer Tumbler",
      "link": "https://shop.googlemerchandisestore.com/store/20230615372/assets/items/images/GGCLCDNJ121099.jpg"
    }
    ===


### Hybrid Search
[Hybrid search](https://weaviate.io/developers/weaviate/search/hybrid) combines keyword (BM25) and vector search together, giving you the best of both algorithms.

To use hybrid search in Weaviate, all you have to do is define the `alpha` parameter to determine the weighting.

`alpha` = 0