# Sub-Question Query Engine with Mistral AI and LlamaIndex 

A `VectorStoreIndex` is adept at addressing queries that pertain to specific contexts within a single document or a collection of documents. However, user queries in real-world scenarios can be intricate, often requiring the retrieval of context from multiple documents to provide an answer. In such situations, a straightforward VectorStoreIndex might not suffice. Instead, breaking down the complex user queries into sub-queries can yield more accurate responses.

In this notebook, we will explore how the `SubQuestionQueryEngine` can be leveraged to tackle complex queries by generating and addressing sub-queries.

### Installation


```python
!pip install llama-index
!pip install llama-index-llms-mistralai
!pip install llama-index-embeddings-mistralai
```

### Setup API Key


```python
import os
os.environ['MISTRAL_API_KEY'] = '<YOUR MISTRAL API KEY>'
```

### Set LLM and Embedding Model


```python
import nest_asyncio

nest_asyncio.apply()
```


```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
```


```python
llm = MistralAI(model='mistral-large')
embed_model = MistralAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

### Logging


```python
# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

import logging
import sys

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Clear out any existing handlers
logger.handlers = []

# Set up the StreamHandler to output to sys.stdout (Colab's output)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

# Add the handler to the logger
logger.addHandler(handler)

from IPython.display import display, HTML
```

### Download Data

We will use `Uber, Lyft 10K SEC Filings` and `Paul Graham Essay Document`.


```python
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O './lyft_2021.pdf'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'
```

### Load Data


```python
# Uber docs
uber_docs = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()

# Lyft docs
lyft_docs = SimpleDirectoryReader(input_files=["./lyft_2021.pdf"]).load_data()

# Paul Graham Essay 
paul_graham_docs = SimpleDirectoryReader(input_files=["./paul_graham_essay.txt"]).load_data()
```

### Index and Query Engine creation


```python
# Index on uber docs
uber_vector_index = VectorStoreIndex.from_documents(uber_docs)

# Index on lyft docs
lyft_vector_index = VectorStoreIndex.from_documents(lyft_docs)

# Index on Paul Graham docs
paul_graham_vector_index = VectorStoreIndex.from_documents(paul_graham_docs)
```

    [HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK"]



```python
# Query Engine over Index with uber docs
uber_vector_query_engine = uber_vector_index.as_query_engine(similarity_top_k = 5)

# Query Engine over Index with lyft docs
lyft_vector_query_engine = lyft_vector_index.as_query_engine(similarity_top_k = 5)

# Query Engine over Index with Paul Graham Essay
paul_graham_vector_query_engine = paul_graham_vector_index.as_query_engine(similarity_top_k = 5)
```

### Create Tools


```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=uber_vector_query_engine,
        metadata=ToolMetadata(
            name="uber_vector_query_engine",
            description=(
                "Provides information about Uber financials for year 2021."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_vector_query_engine,
        metadata=ToolMetadata(
            name="lyft_vector_query_engine",
            description=(
                "Provides information about Lyft financials for year 2021."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=paul_graham_vector_query_engine,
        metadata=ToolMetadata(
            name="paul_graham_vector_query_engine",
            description=(
                "Provides information about paul graham."
            ),
        ),
    ),
]
```

### Create SubQuestion Query Engine


```python
sub_question_query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
```

### Querying

Here you can see the sub-queries created to answer complex user-query which has multiple questions.

#### Query related to Uber and Lyft docs.

Creates two sub-queries related to Uber and Lyft.


```python
response = sub_question_query_engine.query("Compare the revenue of uber and lyft?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"
    Generated 2 sub questions.
    [uber_vector_query_engine] Q: What is the revenue of Uber for year 2021
    [lyft_vector_query_engine] Q: What is the revenue of Lyft for year 2021
    [HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">In 2021, Uber's total revenue was significantly higher than Lyft's. Uber generated $17,455 million from various services like Mobility, Delivery, Freight, and other revenue streams. On the other hand, Lyft's revenue for the same year was $3,208,323 in thousands, which translates to $3,208.323 million. Therefore, Uber's revenue was approximately five times that of Lyft in 2021.</p>


#### Query related to Uber and Paul Graham Essay

Creates two sub-queries related to Uber and Paul Graham Essay.


```python
response = sub_question_query_engine.query("What is the revenue of uber and why did paul graham start YC?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"
    Generated 2 sub questions.
    [uber_vector_query_engine] Q: What is the revenue of Uber for the year 2021
    [paul_graham_vector_query_engine] Q: Why did Paul Graham start Y Combinator
    [HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">Uber's revenue for the year 2021 was reported to be $17,455 million. As for the reason behind Paul Graham starting Y Combinator, it was a result of three main factors. Firstly, he wanted to stop delaying his plans for angel investing. Secondly, he wished to collaborate with Robert and Trevor on projects. Lastly, he was frustrated with venture capitalists who took too long to make decisions. Therefore, he decided to start his own investment firm, where he would fund it, his wife Jessica could work for it, and they could bring Robert and Trevor on board as partners. The goal was to make seed investments and assist startups in the same way they had been helped when they were starting out.</p>


#### Query Related to Uber, Lyft and Paul Graham Essay.

Creates sub-queries related to Uber, Lyft and Paul Graham Essay.


```python
response = sub_question_query_engine.query("Compare revenue of uber with lyft and why did paul graham start YC?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"
    Generated 3 sub questions.
    [uber_vector_query_engine] Q: What is the revenue of Uber for year 2021
    [lyft_vector_query_engine] Q: What is the revenue of Lyft for year 2021
    [paul_graham_vector_query_engine] Q: Why did Paul Graham start Y Combinator
    [HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">In 2021, Uber's total revenue was significantly higher than Lyft's. Uber generated a total of $17,455 million from various offerings like Mobility, Delivery, Freight, and other revenue streams. On the other hand, Lyft's revenue for the same year was $3,208,323 (in thousands).

Paul Graham started Y Combinator due to a combination of factors. Firstly, he wanted to stop delaying his plans to become an angel investor. Secondly, he wished to collaborate with Robert and Trevor on projects. Lastly, he was frustrated with the slow decision-making process of venture capitalists. These factors led him to establish his own investment firm, where he could make seed investments and assist startups in a manner similar to how he had been helped when starting out.</p>