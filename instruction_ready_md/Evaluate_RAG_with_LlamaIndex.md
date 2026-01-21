# Building and Evaluating a RAG Pipeline with LlamaIndex

This guide walks you through constructing a Retrieval-Augmented Generation (RAG) pipeline and systematically evaluating its performance using LlamaIndex's evaluation modules.

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install llama-index pandas
```

## 1. Setup and Imports

First, apply `nest_asyncio` for compatibility in environments like Jupyter, then import the required modules.

```python
import nest_asyncio
nest_asyncio.apply()

from llama_index.evaluation import (
    generate_question_context_pairs,
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
import os
import pandas as pd
```

Set your OpenAI API key as an environment variable.

```python
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
```

## 2. Build the RAG Pipeline

### 2.1 Download and Load Data

We'll use Paul Graham's essay as our sample data.

```bash
mkdir -p 'data/paul_graham/'
curl 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -o 'data/paul_graham/paul_graham_essay.txt'
```

Now, load the documents and build the index.

```python
# Load documents from the data directory
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Define the LLM (GPT-4) for indexing and querying
llm = OpenAI(model="gpt-4")

# Parse documents into nodes with a chunk size of 512
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# Create a vector index from the nodes
vector_index = VectorStoreIndex(nodes)
```

### 2.2 Create a Query Engine

Instantiate a query engine from the index to handle user queries.

```python
query_engine = vector_index.as_query_engine()
```

### 2.3 Test the Query Engine

Let's run a sample query to verify the pipeline works.

```python
response_vector = query_engine.query("What did the author do growing up?")
print(response_vector.response)
```

By default, the retriever fetches the top 2 most relevant nodes. You can inspect the retrieved context.

```python
# Inspect the first retrieved node
print(response_vector.source_nodes[0].get_text())

# Inspect the second retrieved node
print(response_vector.source_nodes[1].get_text())
```

## 3. Evaluate the RAG System

Evaluation is crucial for assessing the quality of your RAG pipeline. We'll evaluate both retrieval accuracy and response quality.

### 3.1 Generate an Evaluation Dataset

First, create a dataset of question-context pairs from your indexed nodes. This dataset will be used for both retrieval and response evaluation.

```python
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=2
)
```

### 3.2 Evaluate Retrieval Performance

We'll evaluate the retriever using **Hit Rate** and **Mean Reciprocal Rank (MRR)**.

*   **Hit Rate:** The fraction of queries where the correct answer is within the top-k retrieved documents.
*   **MRR:** The average reciprocal rank of the highest-placed relevant document across all queries. A higher score indicates more relevant documents are ranked at the top.

```python
# Create a retriever that fetches the top 2 results
retriever = vector_index.as_retriever(similarity_top_k=2)

# Initialize the RetrieverEvaluator with the desired metrics
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

# Run the evaluation on the generated dataset
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
```

Define a helper function to display the results cleanly.

```python
def display_results(name, eval_results):
    """Display results from evaluate."""
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )
    return metric_df

# Display the evaluation results
results_df = display_results("OpenAI Embedding Retriever", eval_results)
print(results_df)
```

**Observation:** If MRR is lower than the Hit Rate, it indicates that while the correct context is often retrieved, it is not consistently ranked first. Techniques like using a reranker model can help improve the ranking order.

### 3.3 Evaluate Response Quality

Response evaluation focuses on two key aspects:
1.  **Faithfulness:** Is the generated answer grounded in the retrieved context (i.e., not hallucinated)?
2.  **Relevancy:** Does the answer appropriately address the original query?

First, extract the list of queries from our evaluation dataset.

```python
queries = list(qa_dataset.queries.values())
```

#### 3.3.1 Configure LLMs for Evaluation

We'll use GPT-3.5 Turbo to generate responses and GPT-4 to evaluate them for higher accuracy.

```python
# LLM for generating responses
gpt35 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_gpt35 = ServiceContext.from_defaults(llm=gpt35)

# LLM for running evaluations
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

# Recreate the query engine with the GPT-3.5 Turbo service context
vector_index = VectorStoreIndex(nodes, service_context=service_context_gpt35)
query_engine = vector_index.as_query_engine()
```

#### 3.3.2 Faithfulness Evaluation

The `FaithfulnessEvaluator` checks if the response is directly supported by the source nodes.

```python
faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)

# Test on a single query
eval_query = queries[10]
print(f"Evaluation Query: {eval_query}")

response_vector = query_engine.query(eval_query)
eval_result = faithfulness_gpt4.evaluate_response(response=response_vector)

print(f"Passing: {eval_result.passing}")
```

#### 3.3.3 Relevancy Evaluation

The `RelevancyEvaluator` assesses whether the response (and its source context) correctly answers the query.

```python
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)

# Test on a single query
query = queries[10]
response_vector = query_engine.query(query)

eval_result = relevancy_gpt4.evaluate_response(
    query=query, response=response_vector
)

print(f"Passing: {eval_result.passing}")
print(f"Feedback: {eval_result.feedback}")
```

### 3.4 Batch Evaluation

For efficiency, use the `BatchEvalRunner` to compute multiple evaluation metrics across several queries at once.

```python
# Select a subset of queries for batch evaluation
batch_eval_queries = queries[:10]

# Initialize the runner with both evaluators
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8, # Adjust based on your environment
)

# Run the batch evaluation
eval_results = await runner.aevaluate_queries(
    query_engine, queries=batch_eval_queries
)

# Calculate aggregate scores
faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])

print(f"Faithfulness Score: {faithfulness_score}")
print(f"Relevancy Score: {relevancy_score}")
```

**Observation:** Scores of `1.0` indicate perfect faithfulness (no hallucinations) and relevancy (all answers are appropriate to the queries) within the evaluated subset.

## Conclusion

You have successfully built a RAG pipeline using LlamaIndex and evaluated its core components. The retrieval evaluation (Hit Rate, MRR) quantifies how well your system finds relevant context, while the response evaluation (Faithfulness, Relevancy) measures the quality of the final answer.

To dive deeper, explore LlamaIndex's comprehensive [evaluation guide](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/root.html), which includes additional metrics and advanced techniques like using rerankers to improve retrieval order.