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

# Anomaly detection with embeddings

## Overview

This tutorial demonstrates how to use the embeddings from the Gemini API to detect potential outliers in your dataset. You will visualize a subset of the 20 Newsgroup dataset using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and detect outliers outside a particular radius of the central point of each categorical cluster.



```
%pip install -U -q "google-genai>=1.0.0"
```

    [0.0/137.7 kB, ..., 137.7/137.7 kB]

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
# Used to securely store your API key
from google.colab import userdata
from google import genai
API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
```

## Prepare dataset

The [20 Newsgroups Text Dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) from the open-source [SciKit project](https://scikit-learn.org/) contains 18,000 newsgroups posts on 20 topics divided into training and test sets. The split between the training and test datasets are based on messages posted before and after a specific date. This tutorial uses the training subset.


```
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset="train")

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
idx = newsgroups_train.data[0].index("Lines")
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
import re

# Apply functions to remove names, emails, and extraneous words from data points in newsgroups.data
newsgroups_train.data = [
    re.sub(r"[\w\.-]+@[\w\.-]+", "", d) for d in newsgroups_train.data
]  # Remove email
newsgroups_train.data = [
    re.sub(r"\([^()]*\)", "", d) for d in newsgroups_train.data
]  # Remove names
newsgroups_train.data = [
    d.replace("From: ", "") for d in newsgroups_train.data
]  # Remove "From: "
newsgroups_train.data = [
    d.replace("\nSubject: ", "") for d in newsgroups_train.data
]  # Remove "\nSubject: "

# Cut off each text entry after 5,000 characters
newsgroups_train.data = [
    d[0:5000] if len(d) > 5000 else d for d in newsgroups_train.data
]
```


```
import pandas as pd

# Put training points into a dataframe
df_train = pd.DataFrame(newsgroups_train.data, columns=["Text"])
df_train["Label"] = newsgroups_train.target
# Match label to target name index
df_train["Class Name"] = df_train["Label"].map(
    newsgroups_train.target_names.__getitem__
)

df_train
```





  <div id="df-da004801-7c77-418d-b324-70928f41a161" class="colab-df-container">
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
<p>11314 rows × 3 columns</p>
</div>




Next, sample some of the data by taking 150 data points in the training dataset and choosing a few categories. This tutorial uses the science categories.


```
# Take a sample of each label category from df_train
SAMPLE_SIZE = 150
df_train = (
    df_train.groupby("Label", as_index=False)
    .apply(lambda x: x.sample(SAMPLE_SIZE))
    .reset_index(drop=True)
)

# Choose categories about science
df_train = df_train[df_train["Class Name"].str.contains("sci")]

# Reset the index
df_train = df_train.reset_index()
df_train
```

    <ipython-input-7-dc22d2141534>:5: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      .apply(lambda x: x.sample(SAMPLE_SIZE))






  <div id="df-c9796cf4-f9fb-4f88-ace7-4d610b27f50a" class="colab-df-container">
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
      <td>Re: White House Public Encryption Management ...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1651</td>
      <td>Re: The [secret] source of that announcement\...</td>
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
      <td>Re: White House Wiretap Chip Disinformation S...</td>
      <td>11</td>
      <td>sci.crypt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1654</td>
      <td>Tony Lezard &lt;&gt;Re: text of White House announce...</td>
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
      <td>Leigh Palmer &lt;&gt;Re: Orion drive in vacuum -- ho...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>596</th>
      <td>2246</td>
      <td>Re: A WRENCH in the works?\nOrganization: Ken...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>597</th>
      <td>2247</td>
      <td>Re: space news from Feb 15 AW&amp;ST\nOrganizatio...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>598</th>
      <td>2248</td>
      <td>Subject: DC-X/Y/1 question\n \nKeywords: DC-X\...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>599</th>
      <td>2249</td>
      <td>MUNIZB% Long Island \nX-Added: Forwarded by Sp...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 4 columns</p>
</div>




```
df_train["Class Name"].value_counts()
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
      <th>count</th>
    </tr>
    <tr>
      <th>Class Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sci.crypt</th>
      <td>150</td>
    </tr>
    <tr>
      <th>sci.electronics</th>
      <td>150</td>
    </tr>
    <tr>
      <th>sci.med</th>
      <td>150</td>
    </tr>
    <tr>
      <th>sci.space</th>
      <td>150</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



## Create the embeddings

In this section, you will see how to generate embeddings for the different texts in the dataframe using the embeddings from the Gemini API.

### API changes to Embeddings with model embedding-001

For the embeddings model, `text-embedding-004`, there is a task type parameter and the optional title (only valid with task_type=`RETRIEVAL_DOCUMENT`).

These parameters apply only to the embeddings models. The task types are:

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

from google.api_core import retry
import numpy as np
import math


def make_embed_text_fn(model):

    @retry.Retry(timeout=300.0)
    def embed_fn(texts: list[str]) -> list[list[float]]:
        # Set the task_type to CLUSTERING and embed the batch of texts
        embeddings = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="CLUSTERING"),
        ).embeddings
        return np.array([embedding.values for embedding in embeddings])

    return embed_fn


def create_embeddings(df):
    MODEL_ID = "text-embedding-004" # @param ["embedding-001","text-embedding-004"] {allow-input: true}
    model = f"models/{MODEL_ID}"
    embed_fn = make_embed_text_fn(model)

    batch_size = 100  # at most 100 requests can be in one batch
    all_embeddings = []

    # Loop over the texts in chunks of batch_size
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df["Text"].iloc[i : i + batch_size].tolist()
        embeddings = embed_fn(batch)
        all_embeddings.extend(embeddings)

    df["Embeddings"] = all_embeddings
    return df


df_train = create_embeddings(df_train)
df_train.drop("index", axis=1, inplace=True)
```


    [0/6, ..., 6/6]

## Dimensionality reduction

The dimension of the document embedding vector is 768. In order to visualize how the embedded documents are grouped together, you will need to apply dimensionality reduction as you can only visualize the embeddings in 2D or 3D space. Contextually similar documents should be closer together in space as opposed to documents that are not as similar.


```
len(df_train["Embeddings"][0])
```




    768




```
# Convert df_train['Embeddings'] Pandas series to a np.array of float32
X = np.array(df_train["Embeddings"].to_list(), dtype=np.float32)
X.shape
```




    (600, 