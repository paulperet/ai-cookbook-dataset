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

# Using Gemini API with Qdrant vector search for hybrid retrieval in legal AI


<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/qdrant/Hybrid_Search_Legal.ipynb"></a>

<!-- Community Contributor Badge -->
<table>
  <tr>
    <!-- Author Avatar Cell -->
    <td bgcolor="#d7e6ff">
      <a href="https://github.com/mrscoopers" target="_blank" title="View Jenny's profile on GitHub">
        
      </a>
    </td>
    <!-- Text Content Cell -->
    <td bgcolor="#d7e6ff">
      <h2>This notebook was contributed by <a href="https://github.com/mrscoopers" target="_blank"><strong>Jenny</strong></a>.</h2>
      <h5><a href="https://www.linkedin.com/in/evgeniya-sukhodolskaya/">Jenny's LinkedIn</a></h5><br>
      <!-- Footer -->
      <small><em>Have a cool Gemini example? Feel free to <a href="https://github.com/google-gemini/cookbook/blob/main/CONTRIBUTING.md" target="_blank">share it too</a>!</em></small>
    </td>
  </tr>
</table>

<!-- Princing warning Badge -->
<table>
  <tr>
    <!-- Emoji -->
    <td bgcolor="#f5949e">
      <font size=30>⚠️</font>
    </td>
    <!-- Text Content Cell -->
    <td bgcolor="#f5949e">
      <h3>This notebook requires paid tier rate limits to run properly.<br>  
(cf. <a href="https://ai.google.dev/pricing#veo2">pricing</a> for more details).</h3>
    </td>
  </tr>
</table>

## Overview


In the legal domain, **accuracy** and **factual correctness** are immensely critical.

A Legal AI startup that collaborated with [Qdrant](https://qdrant.tech/) has outlined **an approach to securing both in Legal AI applications** (for example, Retrieval Augmented Generation (RAG)-based or agentic):

> *“Turn everything into a retrieval problem where you're retrieving ground truth. If you frame it that way, you don't have to worry about hallucinations, as everything given to the user is grounded in some part of a valid document.”*

Truly, many Legal AI businesses require **high-quality retrieval** in their applications. To get there, you need:
- The knowledge of the right tools and techniques that increase search relevance;
- A well-suited embedding model;
- Being ready to experiment!:)

### This notebook

In this notebook, you’ll learn how to combine `gemini-embedding-001` with the tools provided by the Qdrant vector search engine to build a **legal QA retrieval pipeline**.

You'll learn how to:
- Set up a hybrid search (dense + keyword) in Qdrant;
- Use [Matryoshka Representations](https://huggingface.co/blog/matryoshka) of Gemini embeddings to trade off quality vs. cost.

## Setup

### Install SDK

- `google-genai` for `gemini-embedding-001` embeddings;
- `qdrant-client[fastembed]` - the Qdrant's python client;
- HuggingFace `datasets` - to load open sourced legal Q&A datasets


```
%pip install -q -U "google-genai>=1.0.0" qdrant-client[fastembed] datasets
```

### Set up your API keys:

- `GOOGLE_API_KEY`, required for using `gemini-embedding-001` embeddings  
  (look up how to generate it [here](https://ai.google.dev/gemini-api/docs/api-key))

- `QDRANT_API_KEY` and `QDRANT_URL` from a **free-forever** Qdrant Cloud cluster  
(you'll be guided on how to get both in the [Qdrant Cloud UI](https://cloud.qdrant.io/))

To run the following cell, your API keys must be stored in a Colab Secret tab.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
QDRANT_API_KEY = userdata.get('QDRANT_API_KEY')
QDRANT_URL = userdata.get('QDRANT_URL')
```

## Step 1: Download the Dataset

You'll use one of the [Hugging Face datasets from Isaacus](https://huggingface.co/isaacus), a legal artificial intelligence research company.

A common use case in legal AI is a Retrieval-Augmented Generation (RAG) chatbot. To evaluate retrieval performance for such applications, you need a Question-Answer (QA) dataset.

### Choosing a Dataset

- [Open Australian Legal QA](https://huggingface.co/datasets/isaacus/open-australian-legal-qa) looks interesting. However, all its LLM-generated questions mention the exact name of the legal case, which also appears in the answer. The dataset maps each question to one answer (1:1), making it trivial to build a perfect retriever => not even close to real-life scenarios:)

- Instead, let's consider [LegalQAEval](https://huggingface.co/datasets/isaacus/LegalQAEval). It looks more like the kind of questions a user might ask a RAG-based legal chatbot. For example:  
  * "*How are pharmacists regulated in most jurisdictions?*"
  * "*what is ncts*"

#### LegalQAEval

This dataset contains ~2400 QA pairs and includes:

- `id`: a unique string identifier;
- `question`: a natural language question;
- `text`: a chunk of text that *may* contain the answer;
- `answers`: a list of answers (and their positions within the text), or `null` if the `text` does not have the answer.

Load the legal QA corpus; you'll use all available splits.


```
from datasets import load_dataset, concatenate_datasets

corpus = concatenate_datasets(load_dataset('isaacus/LegalQAEval', split=['val', 'test']))
```

### Text chunks deduplication

Since the dataset can contain `text` chunks with multiple questions related to them, initially deduplicate `text` fields to not store identical information several times.


```
import pandas as pd
import datasets

# Convert the Hugging Face dataset to a pandas DataFrame
df = corpus.to_pandas()

# Group by 'text' and aggregate 'id' into a list
grouped_corpus = df.groupby('text')['id'].apply(list).reset_index().rename(columns={'id': 'ids'})

corpus_deduplicated = datasets.Dataset.from_pandas(grouped_corpus)
```

## Step 2: Define the use case configuration

In a typical legal chatbot scenario, users ask a question, and an LLM generates an answer based on a relevant text chunk.

To imitate it, you'll need to store in Qdrant numerical representations (embeddings) of `text` chunks.  
During retrieval, a `question` will be converted into a numerical representation in the same embedding space. Then, (approximately) the nearest `text` chunk will be found in the vector index.

> The Gemini embedding model [supports](https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types) RAG-style Q&A retrieval (task type `QUESTION_ANSWERING`).

Now, to fully define our storage configuration, let's consider several factors relevant to a common RAG use case in the legal AI domain.

### Cost versus accuracy: matryoshka representations

Gemini `gemini-embedding-001` embeddings are 3072-dimensional.  
In a RAG setup with ~1 million chunks, storing such embeddings in RAM (for fast retrieval) would require about **12 GB**.

The Gemini embedding model supports an approach to balance accuracy & cost of retrieval. It is trained using [Matryoshka Representation Learning (MRL)](https://ai.google.dev/gemini-api/docs/embeddings#control-embedding-size), meaning that the most important information about the encoded text is stored in the first dimensions of the embedding.

So, you can, for example:
- Use only the first 768 dimensions of the Gemini embedding for **faster retrieval**;
- And then **rerank** the retrieved results using the full 3072-dimensional embeddings for higher precision.

### Accuracy from the best of both worlds: hybrid search

In legal use cases, it is often beneficial to combine the strengths of:  
- **Keyword-based search (lexical)** for more direct control over matches;  
- **Embedding-based search (semantic)** for handling questions phrased in a conversational way.

> **In Qdrant, both approaches can be combined in [hybrid & multi-stage queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)**.  

For the keyword-based part, Qdrant supports multiple options, from traditional BM25 to sparse neural retrievers like [SPLADE](https://qdrant.tech/documentation/fastembed/fastembed-splade/). Among the options, there's [**our custom improvement of BM25 called miniCOIL**](https://qdrant.tech/documentation/fastembed/fastembed-minicoil/), which you will use in this notebook.

> In Qdrant, keyword-based retrieval is achieved using [sparse vectors](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors).


### Collection configuration
Configure a Qdrant collection for the legal QA retrieval pipeline.


```
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(  # Initializing Qdrant client.
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "legal_AI_QA"
GEMINI_EMBEDDING_RETRIEVAL_SIZE = 768
GEMINI_EMBEDDING_FULL_SIZE = 3072

if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "gemini_embedding_retrieve": models.VectorParams(
                size=GEMINI_EMBEDDING_RETRIEVAL_SIZE,  # Smaller embeddings for faster retrieval.
                distance=models.Distance.COSINE,
            ),
            "gemini_embedding_rerank": models.VectorParams(
                size=GEMINI_EMBEDDING_FULL_SIZE,  # Full-sized embeddings for precision-boosting reranking.
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=0  # Since these embeddings aren't used for retrieval, you don't need to spend resources on building a vector index.
                ),
                on_disk=True,  # To save on RAM used for retrieval.
            ),
        },
        sparse_vectors_config={
            "miniCOIL": models.SparseVectorParams(
                modifier=models.Modifier.IDF  # Inverse Document Frequency statistic, computed on the Qdrant side.
            )
        },
    )
```

## Step 3: embed texts & index data to Qdrant

To speed up the process of converting the data, you'll:

1. Embed with Gemini all `text` chunks in batches using the `get_embeddings_batch` function.  
2. Upload the results to Qdrant in batches.  
The Qdrant Python client provides the functions `upload_collection` and `upload_points`. These handle batching, retries, and parallelization. They take generators as input, so you'll create a generator function `qdrant_points_stream` for this purpose.

> **Note:** Qdrant automatically normalizes uploaded embeddings if the distance function in your collection was set to `COSINE` (cosine similarity). This means you don’t need to pre-normalize truncated Gemini Matryoshka embeddings, as it's [recommended in the Gemini documentation](https://ai.google.dev/gemini-api/docs/embeddings#quality-for-smaller-dimensions).



```
GEMINI_MODEL_ID = "gemini-embedding-001" # @param ["gemini-embedding-001"] {"allow-input":true, isTemplate: true}
```


```
from google import genai
from google.genai import types
from google.api_core import retry
import uuid

google_client = genai.Client(api_key=GOOGLE_API_KEY)

@retry.Retry(timeout=300)
def get_embeddings_batch(texts, task_type: str = "RETRIEVAL_DOCUMENT"):
    """Generates embeddings for a batch of texts.

    Args:
        texts: A list of strings to embed.
        task_type: The task type for the embedding model.

    Returns:
        A list of embedding vectors.

    Raises:
        Exception: If an error occurs during embedding generation.
    """
    try:
        res = google_client.models.embed_content(
            model=GEMINI_MODEL_ID,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in res.embeddings]
    except Exception as e:
        print(f"An error occurred while getting embeddings: {e}")
        raise


def qdrant_points_stream(corpus, avg_corpus_text_length, gemini_batch_size: int = 8):
    """Streams Qdrant points with embeddings for a given corpus.

    Args:
        corpus: The dataset to process.
        avg_corpus_text_length: The average text length for miniCOIL (based on BM25 formula).
        gemini_batch_size: The batch size for Gemini embedding requests.

    Yields:
        Qdrant PointStruct objects.
    """
    for start in range(0, len(corpus), gemini_batch_size):  # Iterate over the dataset in batches.
        end = min(start + gemini_batch_size, len(corpus))
        batch = corpus.select(range(start, end))  # Current batch slice.

        gemini_embeddings_full = get_embeddings_batch(
            [row["text"] for row in batch], task_type="RETRIEVAL_DOCUMENT"
            )  # Generate embeddings for this batch.

        for batch_item, gemini_embedding_full in zip(batch, gemini_embeddings_full):
            yield models.PointStruct(
                id=str(uuid.uuid4()),  # Unique ID (string UUID or integer supported by Qdrant).
                payload={  # Metadata stored alongside the vector.
                    "text": batch_item["text"],  # Raw text for users/LLMs.
                    "ids": batch_item["ids"],  # IDs of the QA pairs related to this `text` (for later evaluation).
                },
                vector={  # Embeddings.
                    "gemini_embedding_rerank": gemini_embedding_full,  # Full Gemini embedding for reranking.
                    "gemini_embedding_retrieve": gemini_embedding_full[:768],  # Truncated Gemini embedding for retrieval.
                    "miniCOIL": models.Document(  # Custom Qdrant-optimized BM25 replacement.
                        text=batch_item["text"],
                        model="Qdrant/minicoil-v1",
                        options={"avg_len": avg_corpus_text_length, "k": 0.9, "b": 0.4},  # Corpus avg length, k_1 & b from BM25 formula.
                    ),
                },
            )

```

Now you'll embed the data and upload the embeddings.

> Try experimenting with different batch sizes when generating embeddings and uploading them to Qdrant.  
The fastest setup usually depends on your network speed & RAM/CPU/GPU, and keep in mind that embedding inference is not a very fast process.

> The representations used in Qdrant for the keyword-based retrieval part of hybrid search are produced by Qdrant.  
In Colab, Qdrant will download the required models the first time you use them (in our case, **Qdrant/minicoil-v1**), as they’re needed for converting `text` chunks to sparse representations.



```
import tqdm

COLLECTION_NAME = "legal_AI_QA"

# Estimating the average length of the texts in the corpus on the subsample of 1000, to use in BM25-inspired keywords-based retrieval.
SUBSET_SIZE = 1000
avg_corpus_text_length = sum(len(text.split()) for text in corpus["text"][:SUBSET_SIZE]) / SUBSET_SIZE

qdrant_client.upload_points(
    collection_name=COLLECTION_NAME,
    points=tqdm.tqdm(
        qdrant_points_stream(corpus_deduplicated,
                            avg_corpus_text_length=avg_corpus_text_length,
                            gemini_batch_size=4),
        desc="Uploading points",
    ),
    batch_size=4,
)
```

## Step 4: experiment & evaluate

What’s important for every retrieval task is experimenting with different instruments & running evaluations based on a sensible metric.

### Metric
In RAG, the goal is usually to get the **correct result within the top-N retrieved results, using a very small N**, since that’s what the LLM will use to generate a grounded answer, and you'd want to save context window size/reduce token costs.  

You'll use the metric **`hit@1`**, meaning the top-1 ranked text chunk is actually the answer to the question.

### Eval set
For experiments, you should only use questions where the `answers` field is not `null`, since this guarantees that this text chunk contains the answer to the question.



```
questions = corpus.filter(lambda item: len(item['answers']) > 0)
```

Inference Gemini embeddings for all the questions, so you can experiment freely without spending extra time or money.


```
import tqdm

question_embeddings = {}

question_texts = [q['question'] for q in questions]
question_ids = [q['id'] for q in questions]
all_embeddings = []

BATCH_SIZE = 32

for i in tqdm.t