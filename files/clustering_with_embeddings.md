##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Clustering with embeddings

## Overview

This tutorial demonstrates how to visualize and perform clustering with the embeddings from the Gemini API. You will visualize a subset of the 20 Newsgroup dataset using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and cluster that subset using the KMeans algorithm.

For more information on getting started with embeddings generated from the Gemini API, check out the [Get Started](../quickstarts/Get_started.ipynb).

## Prerequisites

You can run this quickstart in Google Colab.

To complete this quickstart on your own development environment, ensure that your envirmonement meets the following requirements:

-  Python 3.11+
-  An installation of `jupyter` to run the notebook.

## Setup

First, download and install the Gemini API Python library.


```
!pip install -U -q google-genai
```


```
import re
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google import genai
from google.genai import types

# Used to securely store your API key
from google.colab import userdata

from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

### Grab an API Key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in Google AI Studio.

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

**Note**: At this time, the Gemini API is [only available in certain regions](https://ai.google.dev/gemini-api/docs/available-regions).


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

## Dataset

The [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) contains 18,000 newsgroups posts on 20 topics divided into training and test sets. The split between the training and test datasets are based on messages posted before and after a specific date. For this tutorial, you will be using the training subset.


```
newsgroups_train = fetch_20newsgroups(subset='train')

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



Here is the first example in the training set.


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
    
    
    
    
    



```
# Apply functions to remove names, emails, and extraneous words from data points in newsgroups.data
newsgroups_train.data = [re.sub(r'[\w\.-]+@[\w\.-]+', '', d) for d in newsgroups_train.data] # Remove email
newsgroups_train.data = [re.sub(r"\([^()]*\)", "", d) for d in newsgroups_train.data] # Remove names
newsgroups_train.data = [d.replace("From: ", "") for d in newsgroups_train.data] # Remove "From: "
newsgroups_train.data = [d.replace("\nSubject: ", "") for d in newsgroups_train.data] # Remove "\nSubject: "
```


```
# Put training points into a dataframe
df_train = pd.DataFrame(newsgroups_train.data, columns=['Text'])
df_train['Label'] = newsgroups_train.target
# Match label to target name index
df_train['Class Name'] = df_train['Label'].map(newsgroups_train.target_names.__getitem__)
# Retain text samples that can be used in the gecko model.
df_train = df_train[df_train['Text'].str.len() < 10000]

df_train
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
      <th>Text</th>
      <th>Label</th>
      <th>Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WHAT car is this!?\nNntp-Posting-Host: rac3.w...</td>
      <td>7</td>
      <td>rec.autos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SI Clock Poll - Final Call\nSummary: Final ca...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PB questions...\nOrganization: Purdue Univers...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Re: Weitek P9000 ?\nOrganization: Harris Comp...</td>
      <td>1</td>
      <td>comp.graphics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Re: Shuttle Launch Question\nOrganization: Sm...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11309</th>
      <td>Re: Migraines and scans\nDistribution: world...</td>
      <td>13</td>
      <td>sci.med</td>
    </tr>
    <tr>
      <th>11310</th>
      <td>Screen Death: Mac Plus/512\nLines: 22\nOrganiz...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>11311</th>
      <td>Mounting CPU Cooler in vertical case\nOrganiz...</td>
      <td>3</td>
      <td>comp.sys.ibm.pc.hardware</td>
    </tr>
    <tr>
      <th>11312</th>
      <td>Re: Sphere from 4 points?\nOrganization: Cent...</td>
      <td>1</td>
      <td>comp.graphics</td>
    </tr>
    <tr>
      <th>11313</th>
      <td>stolen CBR900RR\nOrganization: California Ins...</td>
      <td>8</td>
      <td>rec.motorcycles</td>
    </tr>
  </tbody>
</table>
<p>11141 rows × 3 columns</p>
</div>



Next, you will sample some of the data by taking 100 data points in the training dataset, and dropping a few of the categories to run through this tutorial. Choose the science categories to compare.


```
# Take a sample of each label category from df_train
SAMPLE_SIZE = 150
df_train = (df_train.groupby('Label', as_index = False)
                    .apply(lambda x: x.sample(SAMPLE_SIZE))
                    .reset_index(drop=True))

# Choose categories about science
df_train = df_train[df_train['Class Name'].str.contains('sci')]

# Reset the index
df_train = df_train.reset_index()
df_train
```

    /tmp/ipykernel_200977/406673449.py:4: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      .apply(lambda x: x.sample(SAMPLE_SIZE))





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
      <th>index</th>
      <th>Text</th>
      <th>Label</th>
      <th>Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1650</td>
      <td>(Stephan Neuhaus )Re: PGP 2.2: general commen...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1651</td>
      <td>Re: Off the shelf cheap DES keyseach machine ...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1652</td>
      <td>Re: text of White House announcement and Q&amp;As...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1653</td>
      <td>Need help !!\nKeywords: Firewall gateway mode...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1654</td>
      <td>Re: text of White House announcement and Q&amp;As...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>2245</td>
      <td>Re: Space Station Redesign, JSC Alternative #...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>596</th>
      <td>2246</td>
      <td>Portuguese Launch Complex \nOrganization: NAS...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>597</th>
      <td>2247</td>
      <td>Subject: &lt;None&gt;\n\nOrganization: University of...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>598</th>
      <td>2248</td>
      <td>Re: Vandalizing the sky.\nNntp-Posting-Host: ...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>599</th>
      <td>2249</td>
      <td>Re: Crazy? or just Imaginitive?\nOrganization...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 4 columns</p>
</div>




```
df_train['Class Name'].value_counts()
```




    Class Name
    sci.crypt          150
    sci.electronics    150
    sci.med            150
    sci.space          150
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
tqdm.pandas()

from google.api_core import retry

def make_embed_text_fn(model):

  @retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLUSTERING.
    result = client.models.embed_content(model=model,
                                            contents=text,
                                            config=types.EmbedContentConfig(
                                                task_type="clustering"))
    return np.array(result.embeddings[0].values)

  return embed_fn

def create_embeddings(df):
  df['Embeddings'] = df['Text'].progress_apply(make_embed_text_fn(MODEL_ID))
  return df

df_train = create_embeddings(df_train)
```

    [First Entry, ..., Last Entry]


## Dimensionality reduction

The dimension of the document embedding vector is 3072. In order to visualize how the embedded documents are grouped together, you will need to apply dimensionality reduction as you can only visualize the embeddings in 2D or 3D space. Contextually similar documents should be closer together in space as opposed to documents that are not as similar.


```
len(df_train['Embeddings'][0])
```




    3072




```
# Convert df_train['Embeddings'] Pandas series to a np.array of float32
X = np.array(df_train['Embeddings'].to_list(), dtype=np.float32)
X.shape
```




    (600, 3072)



You will apply the t-Distributed Stochastic Neighbor Embedding (t-SNE) approach to perform dimensionality reduction. This technique reduces the number of dimensions, while preserving clusters (points that are close together stay close together). For the original data, the model tries to construct a distribution over which other data points are "neighbors" (e.g., they share a similar meaning). It then optimizes an objective function to keep a similar distribution in the visualization.


```
tsne = TSNE(random_state=0)
tsne_results = tsne.fit_transform(X)
```


```
