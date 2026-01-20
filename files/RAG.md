<a href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/third_party/LlamaIndex/RAG.ipynb" target="_parent"></a>

# RAG Pipeline with LlamaIndex

In this notebook we will look into building RAG with LlamaIndex using `MistralAI LLM and Embedding Model`. Additionally, we will look into using Index as Retreiver.

1. Basic RAG pipeline.
2. Index as Retriever.


```python
!pip install llama-index 
!pip install llama-index-embeddings-mistralai
!pip install llama-index-llms-mistralai
```

## Setup API Keys


```python
import os
os.environ['MISTRAL_API_KEY'] = '<YOUR MISTRALAI API KEY>'
```

## Basic RAG pipeline

Following are the steps involved in Builiding a basic RAG pipeline.

1. Setup LLM and Embedding Model
2. Download Data
3. Load Data
4. Create Nodes
5. Create Index
6. Create Query Engine
7. Querying

Query Engine combines `Retrieval` and `Response Synthesis` modules to generate response for the given query.

### Setup LLM and Embedding Model


```python
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
```


```python
llm = MistralAI(model='mistral-large')
embed_model = MistralAIEmbedding()
```


```python
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model
```

### Download Data

We will use `Uber 2021 10K SEC Filings` for the demonstration.


```python
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf'
```

[First Entry, ..., Last Entry]

### Load Data


```python
from llama_index.core import SimpleDirectoryReader
```


```python
documents = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()
```

### Create Nodes


```python
from llama_index.core.node_parser import TokenTextSplitter
```


```python
splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=0,
)

nodes = splitter.get_nodes_from_documents(documents)
```

### Create Index


```python
from llama_index.core import VectorStoreIndex
```


```python
index = VectorStoreIndex(nodes)
```

### Create Query Engine


```python
query_engine = index.as_query_engine(similarity_top_k=2)
```

### Querying


```python
response = query_engine.query("What is the revenue of Uber in 2021?")
```


```python
print(response)
```

    The total revenue for Uber in 2021 was $17,455 million. This includes revenue from various offerings such as Mobility, Delivery, Freight, and All Other revenue streams. The Mobility revenue was $6,953 million, Delivery revenue was $8,362 million, Freight revenue was $2,132 million, and All Other revenue was $8 million.


## Index as Retriever

We can make use of created index as a `Retriever`. Retriever helps you to retrieve relevant chunks/ nodes for the given user query.

### Create Retriever


```python
retriever = index.as_retriever(similarity_top_k = 2)
```

### Retrieve relevant nodes for a Query


```python
retrieved_nodes = retriever.retrieve("What is the revenue of Uber in 2021?")
```


```python
from llama_index.core.response.notebook_utils import display_source_node

for node in retrieved_nodes:
    display_source_node(node, source_length=1000)
```


**Node ID:** 96264fe5-bc88-4cf8-8905-a9691c39a5c9**Similarity:** 0.8679960824403077**Text:** Note 2 – RevenueThe
 following  tables  present  our  revenues  disaggregated  by  offering  and  geographical  region.  Revenue  by  geographical  region  is  based  on  where  thetransaction
 occurred. This level of disaggregation takes into consideration how the nature, amount, timing, and uncertainty of revenue and cash flows are affectedby economic factors. Revenue 
is presented in the following tables for the years ended December 31, 2019, 2020 and 2021, respectively (in millions):Year Ended December 31,
2019
2020 2021 Mobility revenue 
$ 10,707 $ 6,089 $ 6,953 Delivery revenue 
1,401 3,904 8,362 Freight revenue
731 1,011 2,132 All Other revenue
161 135 8 Total revenue
$ 13,000 $ 11,139 $ 17,455  We
 offer subscription memberships to end-users including Uber One, Uber Pass, Rides Pass, and Eats Pass (“Subscription”). We recognize Subscriptionfees
 ratably over the life of the pass. We allocate Subscription fees earned to Mobility and Delivery revenue on a proportional basis, b...



**Node ID:** 653f0be9-ecfc-4fac-9488-afbd65f44ef2**Similarity:** 0.8590901940430307**Text:** COVID-19.
Revenue
 was $17.5 billion, or up 57% year-over-year, reflecting the overall growth in our Delivery business and an increase in Freight revenue attributable tothe
 acquisition of Transplace in the fourth quarter of 2021 as well as growth in the number of shippers and carriers on the network combined with an increase involumes with our top shippers.
Net
 loss attributable to Uber Technologies, Inc. was $496 million, a 93% improvement year-over-year, driven by a $1.6 billion pre-tax gain on the sale of ourATG
 Business to Aurora, a $1.6 billion pre-tax  net benefit relating to Uber’s equity investments, as  well as reductions in our fixed cost structure and increasedvariable cost effi
ciencies. Net loss attributable to Uber Technologies, Inc. also included $1.2 billion of stock-based compensation expense.Adjusted
 EBITDA loss was $774 million, improving $1.8 billion from 2020 with Mobility Adjusted EBITDA profit of $1.6 billion. Additionally, DeliveryAdjusted
 EBITDA loss of...


```python

```