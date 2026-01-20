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

# Gemini API: Embedding Quickstart with REST

This notebook provides quick code examples that show you how to get started generating embeddings using `curl`.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata
```


```
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
```

## Embed content

Call the `embed_content` method with the `gemini-embedding-001` model to generate text embeddings:


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/gemini-embedding-001",
    "content": {
    "parts":[{
      "text": "Hello world"}]}, }' 2> /dev/null | head
```

    {
      "embedding": {
        "values": [
          -0.02342152,
          0.01676572,
          0.009261323,
          -0.06383,
          -0.0026262768,
          0.0010187156,
          -0.01125684,


# Batch embed content

You can embed a list of multiple prompts with one API call for efficiency.



```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"requests": [{
      "model": "models/gemini-embedding-001",
      "content": {
      "parts":[{
        "text": "What is the meaning of life?"}]}, },
      {
      "model": "models/gemini-embedding-001",
      "content": {
      "parts":[{
        "text": "How much wood would a woodchuck chuck?"}]}, },
      {
      "model": "models/gemini-embedding-001",
      "content": {
      "parts":[{
        "text": "How does the brain work?"}]}, }, ]}' 2> /dev/null | grep -C 5 values
```

    {
      "embeddings": [
        {
          "values": [
            -0.022374554,
            -0.004560777,
            0.013309286,
            -0.0545072,
            -0.02090443,
    --
            0.018649898,
            0.01224912
          ]
        },
        {
          "values": [
            -0.007975887,
            -0.02141119,
            -0.0016711014,
            -0.061006967,
            -0.010629714,
    --
            -0.016098795,
            -0.0049570287
          ]
        },
        {
          "values": [
            -0.0047850125,
            0.008764064,
            0.0062852204,
            -0.017785408,
            -0.02952513,


## Set the output dimensionality
If you're using `gemini-embedding-001`, you can set the `output_dimensionality` parameter to create smaller embeddings.

* `output_dimensionality` truncates the embedding (e.g., `[1, 3, 5]` becomes `[1,3]` when `output_dimensionality=2`).



```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/gemini-embedding-001",
    "output_dimensionality":256,
    "content": {
    "parts":[{
      "text": "Hello world"}]}, }' 2> /dev/null | head
```

    {
      "embedding": {
        "values": [
          -0.02342152,
          0.01676572,
          0.009261323,
          -0.06383,
          -0.0026262768,
          0.0010187156,
          -0.01125684,


## Use `task_type` to provide a hint to the model how you'll use the embeddings

Let's look at all the parameters the embed_content method takes. There are four:

* `model`: Required. Must be `models/gemini-embedding-001` or `models/text-embeddings-004`.
* `content`: Required. The content that you would like to embed.
* `task_type`: Optional. The task type for which the embeddings will be used. See below for possible values.

`task_type` is an optional parameter that provides a hint to the API about how you intend to use the embeddings in your application.

The following task_type parameters are accepted:

* `RETRIEVAL_QUERY` : The given text is a query in a search/retrieval setting.
* `RETRIEVAL_DOCUMENT`: The given text is a document from the corpus being searched.
* `SEMANTIC_SIMILARITY`: The given text will be used for Semantic Textual Similarity (STS).
* `CLASSIFICATION`: The given text will be classified.
* `CLUSTERING`: The embeddings will be used for clustering.



```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/gemini-embedding-001",
    "content": {
    "parts":[{
      "text": "Hello world"}]},
    "task_type": "RETRIEVAL_DOCUMENT" 
    }' 2> /dev/null | head
```

    {
      "embedding": {
        "values": [
          -0.026189324,
          0.003135996,
          0.01789595,
          -0.08768083,
          -0.0034239523,
          0.018811652,
          -0.0107915355,


## Learning more

* Learn more about `gemini-embedding-001` [here](https://ai.google.dev/gemini-api/docs/embeddings).
*   Explore more examples in the [cookbook](https://github.com/google-gemini/cookbook).