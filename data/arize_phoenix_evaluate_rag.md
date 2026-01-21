# Guide: Building and Evaluating a RAG Pipeline with Mistral AI and Phoenix

This guide walks you through building a Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex and Mistral AI, then evaluating its performance with Phoenix Evals for observability.

## Prerequisites

Ensure you have a Mistral AI API key. You'll be prompted to enter it during setup.

## Setup

First, install the required packages.

```bash
pip install -qq arize-phoenix gcsfs nest_asyncio openinference-instrumentation-llama_index
pip install -q llama-index-embeddings-mistralai
pip install -q llama-index-llms-mistralai
pip install -qq "mistralai>=1.0.0"
```

Now, import the necessary libraries and configure your environment.

```python
import os
from getpass import getpass
import pandas as pd
import phoenix as px
from phoenix.otel import register
from mistralai import Mistral
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import nest_asyncio

# Required for async operations in notebooks
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

# Configure pandas for better display
pd.set_option("display.max_colwidth", None)
```

## Step 1: Configure Mistral AI API

Set your Mistral AI API key as an environment variable.

```python
if not (api_key := os.getenv("MISTRAL_API_KEY")):
    api_key = getpass("🔑 Enter your Mistral AI API key: ")
os.environ["MISTRAL_API_KEY"] = api_key
client = Mistral(api_key=api_key)
```

## Step 2: Launch Phoenix and Enable Tracing

Phoenix provides observability by capturing traces of your LLM application. Start the Phoenix session and instrument LlamaIndex to send traces.

```python
# Launch the Phoenix UI
session = px.launch_app()

# Enable OpenInference tracing for LlamaIndex
tracer_provider = register()
LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)
```

You can view the traces by visiting the URL printed after launching the app.

## Step 3: Load Data and Build the Vector Index

We'll use Paul Graham's essay "What I Worked On" as our sample data.

```python
import tempfile
from urllib.request import urlretrieve

# Download the essay
with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(
        "https://raw.githubusercontent.com/Arize-ai/phoenix-assets/main/data/paul_graham/paul_graham_essay.txt",
        tf.name,
    )
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
```

Now, configure the LLM and embedding model, then create a vector index from the documents.

```python
# Define the LLM and embedding model
llm = MistralAI()
embed_model = MistralAIEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model

# Parse documents into nodes and create the index
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)
```

## Step 4: Create a Query Engine and Test Queries

Create a query engine from the index and run a sample query.

```python
query_engine = vector_index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
print(response.response)
```

## Step 5: Simulate Application Usage with a Set of Queries

To evaluate the pipeline, execute a predefined list of questions. This will generate traces in Phoenix.

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

## Step 6: Evaluate Retrieval Performance

First, extract the retrieved documents from the Phoenix traces.

```python
from phoenix.session.evaluation import get_retrieved_documents

retrieved_documents_df = get_retrieved_documents(px.Client())
retrieved_documents_df.head()
```

Now, evaluate the relevance of each retrieved document to its corresponding query using Phoenix's LLM Evals.

```python
from phoenix.evals import MistralAIModel, RelevanceEvaluator, run_evals

relevance_evaluator = RelevanceEvaluator(MistralAIModel)

retrieved_documents_relevance_df = run_evals(
    evaluators=[relevance_evaluator],
    dataframe=retrieved_documents_df,
    provide_explanation=True,
    concurrency=20,
)[0]

retrieved_documents_relevance_df.head()
```

Combine the retrieval data with the relevance evaluations for analysis.

```python
documents_with_relevance_df = pd.concat(
    [retrieved_documents_df, retrieved_documents_relevance_df.add_prefix("eval_")], axis=1
)
documents_with_relevance_df.head()
```

## Step 7: Evaluate Response Quality

Extract the question-answer pairs along with their reference context from the traces.

```python
from phoenix.session.evaluation import get_qa_with_reference

qa_with_reference_df = get_qa_with_reference(px.Client())
qa_with_reference_df.head()
```

Evaluate the correctness of the LLM's answers and check for hallucinations.

```python
from phoenix.evals import HallucinationEvaluator, MistralAIModel, QAEvaluator, run_evals

qa_evaluator = QAEvaluator(MistralAIModel())
hallucination_evaluator = HallucinationEvaluator(MistralAIModel())

qa_correctness_eval_df, hallucination_eval_df = run_evals(
    evaluators=[qa_evaluator, hallucination_evaluator],
    dataframe=qa_with_reference_df,
    provide_explanation=True,
    concurrency=20,
)

qa_correctness_eval_df.head()
hallucination_eval_df.head()
```

## Step 8: Send Evaluations to Phoenix for Visualization

Log all evaluation results to Phoenix to analyze them in the UI.

```python
from phoenix.trace import SpanEvaluations, DocumentEvaluations

px.Client().log_evaluations(
    SpanEvaluations(dataframe=qa_correctness_eval_df, eval_name="Q&A Correctness"),
    SpanEvaluations(dataframe=hallucination_eval_df, eval_name="Hallucination"),
    DocumentEvaluations(dataframe=retrieved_documents_relevance_df, eval_name="relevance"),
)
```

Finally, print the Phoenix URL to view your traces and evaluations.

```python
print("phoenix URL", session.url)
```

## Conclusion

You have successfully built a RAG pipeline using LlamaIndex and Mistral AI, then evaluated its retrieval and response quality using Phoenix Evals. This process allows you to identify weaknesses in your system, such as irrelevant context retrieval or hallucinated responses.

Phoenix offers a wide range of additional evaluations for LLM applications. For more details, refer to the [LLM Evals documentation](https://docs.arize.com/phoenix/llm-evals/llm-evals).