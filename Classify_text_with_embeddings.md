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

# Classify text with embeddings

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

In this notebook, you'll learn to use the embeddings produced by the Gemini API to train a model that can classify different types of newsgroup posts based on the topic.



```
!pip install -U -q google-genai
```

    [First Entry, ..., Last Entry]

### Grab an API Key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in Google AI Studio.

Get an API key

In Colab, add the key to the secrets manager under the "🔑" in the left panel. Give it the name `GEMINI_API_KEY`.

Once you have the API key, pass it to the SDK. You can do this in two ways:

* Put the key in the `GEMINI_API_KEY` environment variable (the SDK will automatically pick it up from there).
* Pass the key to `genai.Client(api_key=...)`


```
from google import genai
from google.genai import types

# Used to securely store your API key
from google.colab import userdata

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

    models/embedding-001
    models/text-embedding-004
    models/gemini-embedding-exp-03-07
    models/gemini-embedding-exp
    models/gemini-embedding-001


### Select the model to be used


```
MODEL_ID = "gemini-embedding-001" # @param ["gemini-embedding-001", "text-embedding-004"] {"allow-input":true, isTemplate: true}
```

## Prepare the dataset

The [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) contains 18,000 newsgroups posts on 20 topics divided into training and test sets. The split between the training and test datasets are based on messages posted before and after a specific date. For this tutorial, you will be using the subsets of the training and test datasets. You will preprocess and organize the data into Pandas dataframes.


```
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# View list of class names for dataset
newsgroups_train.target_names
```




    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']



Here is an example of what a data point from the training set looks like.


```
idx = newsgroups_train.data[0].index('Lines')
print(newsgroups_train.data[0][idx:])
```

    Lines: 15
    
     I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.
    
    Thanks,
    - IL
       ---- brought to you by your neighborhood Lerxst ----
    
    
    
    
    


Now you will begin preprocessing the data for this tutorial. Remove any sensitive information like names, email, or redundant parts of the text like `"From: "` and `"\nSubject: "`. Organize the information into a Pandas dataframe so it is more readable.


```
import re
import pandas as pd

def preprocess_newsgroup_data(newsgroup_dataset):
  # Apply functions to remove names, emails, and extraneous words from data points in newsgroups.data
  newsgroup_dataset.data = [re.sub(r'[\w\.-]+@[\w\.-]+', '', d) for d in newsgroup_dataset.data] # Remove email
  newsgroup_dataset.data = [re.sub(r"\([^()]*\)", "", d) for d in newsgroup_dataset.data] # Remove names
  newsgroup_dataset.data = [d.replace("From: ", "") for d in newsgroup_dataset.data] # Remove "From: "
  newsgroup_dataset.data = [d.replace("\nSubject: ", "") for d in newsgroup_dataset.data] # Remove "\nSubject: "

  # Cut off each text entry after 5,000 characters
  newsgroup_dataset.data = [d[0:5000] if len(d) > 5000 else d for d in newsgroup_dataset.data]

  # Put data points into dataframe
  df_processed = pd.DataFrame(newsgroup_dataset.data, columns=['Text'])
  df_processed['Label'] = newsgroup_dataset.target
  # Match label to target name index
  df_processed['Class Name'] = ''
  for idx, row in df_processed.iterrows():
    df_processed.at[idx, 'Class Name'] = newsgroup_dataset.target_names[row['Label']]

  return df_processed
```


```
# Apply preprocessing function to training and test datasets
df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

df_train.head()
```




| Text | Label | Class Name |
|------|-------|------------|
| 0 | WHAT car is this!?\nNntp-Posting-Host: rac3.w... | 7 | rec.autos |
| 1 | SI Clock Poll - Final Call\nSummary: Final ca... | 4 | comp.sys.mac.hardware |
| 2 | PB questions...\nOrganization: Purdue Univers... | 4 | comp.sys.mac.hardware |
| 3 | Re: Weitek P9000 ?\nOrganization: Harris Comp... | 1 | comp.graphics |
| 4 | Re: Shuttle Launch Question\nOrganization: Sm... | 14 | sci.space |



Next, you will sample some of the data by taking 100 data points in the training dataset, and dropping a few of the categories to run through this tutorial. Choose the science categories to compare.


```
def sample_data(df, num_samples, classes_to_keep):
  df = df.groupby('Label', as_index = False).apply(lambda x: x.sample(num_samples)).reset_index(drop=True)

  df = df[df['Class Name'].str.contains(classes_to_keep)]

  # Reset the encoding of the labels after sampling and dropping certain categories
  df['Class Name'] = df['Class Name'].astype('category')
  df['Encoded Label'] = df['Class Name'].cat.codes

  return df
```


```
TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
CLASSES_TO_KEEP = 'sci' # Class name should contain 'sci' in it to keep science categories
df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)
```

    [First Entry, ..., Last Entry]



```
df_train.value_counts('Class Name')
```




    Class Name
    sci.crypt          100
    sci.electronics    100
    sci.med            100
    sci.space          100
    Name: count, dtype: int64




```
df_test.value_counts('Class Name')
```




    Class Name
    sci.crypt          25
    sci.electronics    25
    sci.med            25
    sci.space          25
    Name: count, dtype: int64



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
from tqdm.auto import tqdm
from google.genai import types

tqdm.pandas()

import numpy as np
import math

def make_embed_text_fn(model):

    def embed_fn(texts: list[str]) -> list[list[float]]:
        # Set the task_type to CLASSIFICATION and embed the batch of texts
        result = client.models.embed_content(model=model,
                                            contents=[texts],
                                            config=types.EmbedContentConfig(
                                                task_type="CLASSIFICATION")).embeddings
        return np.array([embedding.values for embedding in result])

    return embed_fn


def create_embeddings(df):
    embed_fn = make_embed_text_fn(MODEL_ID)

    batch_size = 25  # at most 25 requests can be in one batch
    all_embeddings = []

    # Loop over the texts in chunks of batch_size
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df["Text"].iloc[i : i + batch_size].tolist()
        embeddings = embed_fn(batch)
        all_embeddings.extend(embeddings)

    df["Embeddings"] = all_embeddings
    return df
```


```
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()


def make_embed_text_fn(model):

  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLASSIFICATION.
    result = client.models.embed_content(model=model,
                                            contents=text,
                                            config=types.EmbedContentConfig(
                                                task_type="CLASSIFICATION"))
    return np.array(result.embeddings[0].values)

  return embed_fn

def create_embeddings(df):
  df['Embeddings'] = df['Text'].progress_apply(make_embed_text_fn(MODEL_ID))
  return df
```

Preparing the training dataset:


```
df_train = create_embeddings(df_train)
df_train
```

    100%|██████████| 400/400 [03:35<00:00,  1.86it/s]





| Text | Label | Class Name | Encoded Label | Embeddings |
|------|-------|------------|---------------|------------|
| 1100 | Re: White House Public Encryption Management ... | 11 | sci.crypt | 0 | [-0.01012141, -0.0051694787, 0.0047423732, -0.... |
| 1101 | DES: init vector as additional key?\nKeywords... | 11 | sci.crypt | 0 | [-0.023827964, -0.008963956, -0.0052384217, -0... |
| 1102 | Clipper Chip - How would it work?\nOrganizati... | 11 | sci.crypt | 0 | [-0.027112009, -0.016536575, 0.013272242, -0.0... |
| 1103 | Re: Would "clipper" make a good cover for oth... | 11 | sci.crypt | 0 | [-0.009721769, -0.0026513913, 0.00159029, -0.0... |
| 1104 | * REPORT ON PRIVACY-PROTECTING OFF-LINE CASH ... | 11 | sci.crypt | 0 | [-0.0019142096, -0.01807762, 0.004153099, -0.0... |
| ... | ... | ... | ... | ... | ... |
| 1495 | Re: Biosphere II\nReply-To: \nDisclaimer: Thi... | 14 | sci.space | 3 | [-0.0027465916, -0.009631713, 0.006647122, -0.... |
| 1496 | Re: space news from Feb 1 AW&ST\nNntp-Posting... | 14 | sci.space | 3 | [-0.016138315, 0.011859794, 0.006195726, -0.04... |
| 1497 | Re: Vandalizing the sky.\nIn-Reply-To: 's mes... | 14 | sci.space | 3 | [-0.010342497, -0.004649389, 0.01676999, -0.06... |
| 1498 | Re: Vandalizing the sky.\nOrganization: Unive... | 14 | sci.space | 3 | [-0.008992054, -0.007147352, 0.0055692485, -0.... |
| 1499 | Re: So I'm an idiot, what else is new?\nOrigi... | 14 | sci.space | 3 | [0.003493203, -0.0042175553, -0.0029711807, -0... |
400 rows × 5 columns




```
df_test = create_embeddings(df_test)
df_test
```

    100%|██████████| 100/100 [00:50<00:00,  1.99it/s]





| Text | Label | Class Name | Encoded Label | Embeddings |
|------|-------|------------|---------------|------------|
| 275 | Re: THE CLIPPER CHIP: A TECHNICAL SUMMARY\nOr... | 11 | sci.crypt | 0 | [0.0008749933, -0.009894153, 0.0068213255, -0.... |
| 276 | Amiga RIPEM?\nReply-To: \nOrganization: Emory... | 11 | sci.crypt | 0 | [-0.022581104, -0.0037767778, -0.0035728822, -... |
| 277 | Wiretapping reality today\nOrganization: DSI/... | 11 | sci.crypt | 0 | [-0.0038811539, -0.0055605364, 0.009425233, -0... |
| 278 | Re: What would happen if export restrictions ... | 11 | sci.crypt | 0 | [-0.005810377, 0.0014717537, 0.0027099252, -0.... |
| 279 | Re: Dorothy Denning opposes Clipper, Capstone... | 11 | sci.crypt | 0 | [-0.019840438, -0.013688887, -0.004959918, -0.... |
| ... | ... | ... | ... | ... | ... |
| 370 | Re: No. Re: Space Marketing would be wonderfu... | 14 | sci.space | 3 | [-0.00467501, -0.003431914, 0.024957992, -0.06... |
| 371 | Re: Gamma Ray Bursters.  WHere  are they.\nAr... | 14 | sci.space | 3 | [0.0014332095, 0.0020801027, 0.008766141, -0.0... |
| 372 | Re: Long term Human Missions\nOrganization: S... | 14 | sci.space | 3 | [-0.0030249238, 0.0113083115, 0.027613947, -0.... |
| 373 | Re: Vandalizing the sky.\nOrganization: Joint... | 14 | sci.space | 3 | [-0.0007880609, -0.0064521744, 0.011024299, -0... |
| 374 | Re: DC-X and publicity... is there any ?\nOrg... | 14 | sci.space | 3 | [-0.010754121, 0.002655718, 0.02325094, -0.049... |
100 rows × 5 columns



## Build a simple classification model
Here you will define a simple model with one hidden layer and a single class probability output. The prediction will correspond to the probability of a piece of text being a particular class of news. When you build your model, Keras will automatically shuffle the data points.


```
import keras
from keras import layers

def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
  inputs = x = keras.Input