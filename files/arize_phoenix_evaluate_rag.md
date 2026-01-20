# RAG Observability with Mistral AI and Phoenix

In this tutorial we will look into building a RAG pipeline and evaluating it with Phoenix Evals.

It has the the following sections:

1. Understanding Retrieval Augmented Generation (RAG).
2. Building RAG (with the help of a framework such as LlamaIndex).
3. Evaluating RAG with Phoenix Evals.

## Retrieval Augmented Generation (RAG)

LLMs are trained on vast datasets, but these will not include your specific data (things like company knowledge bases and documentation). Retrieval-Augmented Generation (RAG) addresses this by dynamically incorporating your data as context during the generation process. This is done not by altering the training data of the LLMs but by allowing the model to access and utilize your data in real-time to provide more tailored and contextually relevant responses.

In RAG, your data is loaded and prepared for queries. This process is called indexing. User queries act on this index, which filters your data down to the most relevant context. This context and your query then are sent to the LLM along with a prompt, and the LLM provides a response.

RAG is a critical component for building applications such a chatbots or agents and you will want to know RAG techniques on how to get data into your application.

## Build a RAG system

Now that we have understood the stages of RAG, let's build a pipeline. We will use [LlamaIndex](https://www.llamaindex.ai/) for RAG and [Phoenix Evals](https://docs.arize.com/phoenix/llm-evals/llm-evals) for evaluation.

```python
!pip install -qq arize-phoenix gcsfs nest_asyncio openinference-instrumentation-llama_index
!pip install -q llama-index-embeddings-mistralai
!pip install -q llama-index-llms-mistralai
!pip install -qq "mistralai>=1.0.0"
```

```python
import os
from getpass import getpass
from typing import Any, Dict

import pandas as pd
import phoenix as px
from phoenix.otel import register
from mistralai import Mistral
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

pd.set_option("display.max_colwidth", None)
```

For this tutorial we will be using Mistral to create synthetic data as well as for evaluation.

```python
if not (api_key := os.getenv("MISTRAL_API_KEY")):
    api_key = getpass("🔑 Enter your Mistral AI API key: ")
os.environ["MISTRAL_API_KEY"] = api_key
client = Mistral(api_key=api_key)
```

During this tutorial, we will capture all the data we need to evaluate our RAG pipeline using Phoenix Tracing. To enable this, simply start the phoenix application and instrument LlamaIndex.

```python
session =px.launch_app()
```

Enable Phoenix tracing via `LlamaIndexInstrumentor`. Phoenix uses OpenInference traces - an open-source standard for capturing and storing LLM application traces that enables LLM applications to seamlessly integrate with LLM observability solutions such as Phoenix.

```python
tracer_provider = register()
LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)
```

### Load Data and Build an Index

Let's use an [essay by Paul Graham](https://www.paulgraham.com/worked.html) to build our RAG pipeline.

```python
import tempfile
from urllib.request import urlretrieve

with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(
        "https://raw.githubusercontent.com/Arize-ai/phoenix-assets/main/data/paul_graham/paul_graham_essay.txt",
        tf.name,
    )
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
```

```python
from llama_index.core import Settings

# Define an LLM
llm = MistralAI()
embed_model = MistralAIEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model

# Build index with a chunk_size of 512
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)
```

Build a QueryEngine and start querying.

```python
query_engine = vector_index.as_query_engine()
```

```python
response_vector = query_engine.query("What did the author do growing up?")
```

Check the response that you get from the query.

```python
response_vector.response
```

Remember that we are using Phoenix Tracing to capture all the data we need to evaluate our RAG pipeline. You can view the traces in the phoenix application.

```python
print("phoenix URL", session.url)
```

We have built a RAG pipeline and also have instrumented it using Phoenix Tracing. We now need to evaluate it's performance. We can assess our RAG system/query engine using Phoenix's LLM Evals. Let's examine how to leverage these tools to quantify the quality of our retrieval-augmented generation system.

# Simulate usage of our application

For the evaluation of a RAG system, it's essential to have queries that can fetch the correct context and subsequently generate an appropriate response.

For this tutorial, let's manually generate some questions for our pipeline.

```python
questions_list = [
    "What did the author do growing up?",
    "What was the author's major?",
    "What was the author's minor?",
    "What was the author's favorite class?",
    "What was the author's least favorite class?",
]

for question in questions_list:
    response_vector = query_engine.query(question)
```

# Evaluate our Pipeline

Now that we have executed the queries, we can start validating whether or not the RAG system was able to retrieve the correct context. Let's extract all the retrieved documents from the traces logged to phoenix. (For an in-depth explanation of how to export trace data from the phoenix runtime, consult the [docs](https://docs.arize.com/phoenix/how-to/extract-data-from-spans)).

```python
from phoenix.session.evaluation import get_retrieved_documents

retrieved_documents_df = get_retrieved_documents(px.Client())
retrieved_documents_df
```

Let's now use Phoenix's LLM Evals to evaluate the relevance of the retrieved documents with regards to the query. Note, we've turned on `explanations` which prompts the LLM to explain it's reasoning. This can be useful for debugging and for figuring out potential corrective actions.

```python
from phoenix.evals import (
    MistralAIModel,
    RelevanceEvaluator,
    run_evals,
)

relevance_evaluator = RelevanceEvaluator(MistralAIModel)

retrieved_documents_relevance_df = run_evals(
    evaluators=[relevance_evaluator],
    dataframe=retrieved_documents_df,
    provide_explanation=True,
    concurrency=20,
)[0]
```

```python
retrieved_documents_relevance_df.head()
```

We can now combine the documents with the relevance evaluations to compute retrieval metrics. These metrics will help us understand how well the RAG system is performing.

```python
documents_with_relevance_df = pd.concat(
    [retrieved_documents_df, retrieved_documents_relevance_df.add_prefix("eval_")], axis=1
)
documents_with_relevance_df
```

### Response Evaluation

The retrieval evaluations demonstrates that our RAG system is not perfect. However, it's possible that the LLM is able to generate the correct response even when the context is incorrect. Let's evaluate the responses generated by the LLM.

```python
from phoenix.session.evaluation import get_qa_with_reference

qa_with_reference_df = get_qa_with_reference(px.Client())
qa_with_reference_df
```

Now that we have a dataset of the question, context, and response (input, reference, and output), we now can measure how well the LLM is responding to the queries. For details on the QA correctness evaluation, see the [LLM Evals documentation](https://docs.arize.com/phoenix/llm-evals/running-pre-tested-evals/q-and-a-on-retrieved-data).

```python
from phoenix.evals import (
    HallucinationEvaluator,
    MistralAIModel,
    QAEvaluator,
    run_evals,
)

qa_evaluator = QAEvaluator(MistralAIModel())
hallucination_evaluator = HallucinationEvaluator(MistralAIModel())

qa_correctness_eval_df, hallucination_eval_df = run_evals(
    evaluators=[qa_evaluator, hallucination_evaluator],
    dataframe=qa_with_reference_df,
    provide_explanation=True,
    concurrency=20,
)
```

```python
qa_correctness_eval_df.head()
```

```python
hallucination_eval_df.head()
```

Since we have evaluated our RAG system's QA performance and Hallucinations performance, let's send these evaluations to Phoenix for visualization.

```python
from phoenix.trace import SpanEvaluations, DocumentEvaluations

px.Client().log_evaluations(
    SpanEvaluations(dataframe=qa_correctness_eval_df, eval_name="Q&A Correctness"),
    SpanEvaluations(dataframe=hallucination_eval_df, eval_name="Hallucination"),
    DocumentEvaluations(dataframe=retrieved_documents_relevance_df, eval_name="relevance"),
)
```

We now have sent all our evaluations to Phoenix. Let's go to the Phoenix application and view the results! Since we've sent all the evals to Phoenix, we can analyze the results together to make a determination on whether or not poor retrieval or irrelevant context has an effect on the LLM's ability to generate the correct response.

```python
print("phoenix URL", session.url)
```

## Conclusion

We have explored how to build and evaluate a RAG pipeline using LlamaIndex and Phoenix, with a specific focus on evaluating the retrieval system and generated responses within the pipelines.

Phoenix offers a variety of other evaluations that can be used to assess the performance of your LLM Application. For more details, see the [LLM Evals](https://docs.arize.com/phoenix/llm-evals/llm-evals) documentation.