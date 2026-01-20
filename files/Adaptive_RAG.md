# Adaptive RAG with LlamaIndex

User queries in general can be complex queries, simple queries. One don't always need complex RAG system even to handle simple queries. [Adaptive RAG](https://arxiv.org/abs/2403.14403) proposes an approach to handle complex queries and simple queries.

In this notebook, we will implement an approach similar to Adaptive RAG, which differentiates between handling complex and simple queries. We'll focus on Lyft's 10k SEC filings for the years 2020, 2021, and 2022.

Our approach will involve using `RouterQueryEngine` and `FunctionCalling` capabilities of `MistralAI` to call different tools or indices based on the query's complexity.

- **Complex Queries:** These will leverage multiple tools that require context from several documents.
- **Simple Queries:** These will utilize a single tool that requires context from a single document or directly use an LLM to provide an answer.

Following are the steps we follow here:

1. Download Data.
2. Load Data.
3. Create indices for 3 documents.
4. Create query engines with documents and LLM.
5. Initialize a `FunctionCallingAgentWorker` for complex queries.
6. Create tools.
7. Create `RouterQueryEngine` - To route queries based on its complexity.
8. Querying.

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

### Setup LLM and Embedding Model


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
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
```


```python
# Note: Only `mistral-large-latest` supports function calling
llm = MistralAI(model='mistral-large-latest') 
embed_model = MistralAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model
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

We will download Lyft's 10k SEC filings for the years 2020, 2021, and 2022.


```python
!wget "https://www.dropbox.com/scl/fi/ywc29qvt66s8i97h1taci/lyft-10k-2020.pdf?rlkey=d7bru2jno7398imeirn09fey5&dl=0" -q -O ./lyft_10k_2020.pdf
!wget "https://www.dropbox.com/scl/fi/lpmmki7a9a14s1l5ef7ep/lyft-10k-2021.pdf?rlkey=ud5cwlfotrii6r5jjag1o3hvm&dl=0" -q -O ./lyft_10k_2021.pdf
!wget "https://www.dropbox.com/scl/fi/iffbbnbw9h7shqnnot5es/lyft-10k-2022.pdf?rlkey=grkdgxcrib60oegtp4jn8hpl8&dl=0" -q -O ./lyft_10k_2022.pdf
```

### Load Data


```python
# Lyft 2020 docs
lyft_2020_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2020.pdf"]).load_data()

# Lyft 2021 docs
lyft_2021_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2021.pdf"]).load_data()

# Lyft 2022 docs
lyft_2022_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2022.pdf"]).load_data()
```

### Create Indicies


```python
# Index on Lyft 2020 Document
lyft_2020_index = VectorStoreIndex.from_documents(lyft_2020_docs)

# Index on Lyft 2021 Document
lyft_2021_index = VectorStoreIndex.from_documents(lyft_2021_docs)

# Index on Lyft 2022 Document
lyft_2022_index = VectorStoreIndex.from_documents(lyft_2022_docs)
```

[HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK"]

### Create Query Engines


```python
# Query Engine on Lyft 2020 Docs Index
lyft_2020_query_engine = lyft_2020_index.as_query_engine(similarity_top_k=5)

# Query Engine on Lyft 2021 Docs Index
lyft_2021_query_engine = lyft_2021_index.as_query_engine(similarity_top_k=5)

# Query Engine on Lyft 2022 Docs Index
lyft_2022_query_engine = lyft_2022_index.as_query_engine(similarity_top_k=5)
```

Query Engine for LLM. With this we will use LLM to answer the query.


```python
from llama_index.core.query_engine import CustomQueryEngine

class LLMQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    llm: llm

    def custom_query(self, query_str: str):

        response = self.llm.complete(query_str)

        return str(response)

llm_query_engine = LLMQueryEngine(llm=llm)
```

### Initialize a `FunctionCallingAgentWorker`


```python
# These tools are used to answer complex queries involving multiple documents.
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_2020_query_engine,
        metadata=ToolMetadata(
            name="lyft_2020_10k_form",
            description="Annual report of Lyft's financial activities in 2020",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2021_query_engine,
        metadata=ToolMetadata(
            name="lyft_2021_10k_form",
            description="Annual report of Lyft's financial activities in 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2022_query_engine,
        metadata=ToolMetadata(
            name="lyft_2022_10k_form",
            description="Annual report of Lyft's financial activities in 2022",
        ),
    )
]
```


```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = AgentRunner(agent_worker)
```

### Create Tools

We will create tools using the `QueryEngines`, and `FunctionCallingAgentWorker` created earlier.


```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_2020_query_engine,
        metadata=ToolMetadata(
            name="lyft_2020_10k_form",
            description="Queries related to only 2020 Lyft's financial activities.",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2021_query_engine,
        metadata=ToolMetadata(
            name="lyft_2021_10k_form",
            description="Queries related to only 2021 Lyft's financial activities.",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2022_query_engine,
        metadata=ToolMetadata(
            name="lyft_2022_10k_form",
            description="Queries related to only 2022 Lyft's financial activities.",
        ),
    ),
    QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name="lyft_2020_2021_2022_10k_form",
            description=(
                "Useful for queries that span multiple years from 2020 to 2022 for Lyft's financial activities."

            )
        )
    ),
    QueryEngineTool(
        query_engine=llm_query_engine,
        metadata=ToolMetadata(
            name="general_queries",
            description=(
                "Provides information about general queries other than lyft."
            )
        )
    )
]
```

### Create RouterQueryEngine

`RouterQueryEngine` will route user queries to select one of the tools based on the complexity of the query.


```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose = True
)
```

### Querying

#### Simple Queries:

##### Query: What is the capital of France?

You can see that it used LLM tool since it is a general query.


```python
response = query_engine.query("What is the capital of France?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

[HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", Selecting query engine 4: This option is the most relevant as it mentions 'general queries other than Lyft'. The question about the capital of France is not related to Lyft's financial activities.., HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">The capital of France is Paris. Known as the "City of Light," Paris is famous for its iconic landmarks like the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and more. It is a global center for art, fashion, gastronomy, and culture.</p>


##### Query: What did Lyft do in R&D in 2022?

You can see that it used lyft_2022 tool to answer the query.


```python
response = query_engine.query("What did Lyft do in R&D in 2022?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

[HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", Selecting query engine 2: This choice is most relevant to the question as it pertains to Lyft's financial activities in 2022, which could include information about their Research and Development (R&D) expenditures for that year.., HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">In 2022, Lyft made focused investments in research and development, though the specific details of these investments are not provided in the context. These investments are part of their ongoing commitment to launch new innovations on their platform. Additionally, Lyft completed multiple strategic acquisitions, one of which is PBSC Urban Solutions. This acquisition added over 100,000 bikes to their bikeshare systems across 46 markets in 15 countries.</p>


##### Query: What did Lyft do in R&D in 2021?

You can see that it used lyft_2021 tool to answer the query.


```python
response = query_engine.query("What did Lyft do in R&D in 2021?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

[HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", Selecting query engine 1: This choice is most relevant to the question as it pertains to queries related to only 2021 Lyft's financial activities, which could potentially include information about Lyft's R&D activities during that year.., HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">In 2021, Lyft's research and development expenses increased slightly by $2.8 million. This increase was primarily due to a $51.6 million rise in stock-based compensation. However, there was also a $25.4 million benefit from a restructuring event in the second quarter of 2020, which included a stock-based compensation benefit and severance and benefits costs that did not recur in 2021. These increases were offset by a $37.5 million decrease in personnel-related costs and a $4.6 million decrease in autonomous vehicle research costs.</p>


##### Query: What did Lyft do in R&D in 2020?

You can see that it used lyft_2020 tool to answer the query.


```python
response = query_engine.query("What did Lyft do in R&D in 2020?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

[HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", Selecting query engine 0: This choice is most relevant to the question as it pertains to queries related to only 2020 Lyft's financial activities, which could potentially include information about Lyft's R&D activities during that year.., HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">In 2020, Lyft continued to invest heavily in research and development as part of its mission to revolutionize transportation. The company made focused investments to launch new innovations on its platform. One notable example is the acquisition of Flexdrive, LLC, a longstanding partner in Lyft's Express Drive program. This acquisition is expected to contribute to the growth of Lyft's business and help expand the range of its use cases. Additionally, Lyft invested in the expansion of its network of Light Vehicles and autonomous open platform technology to remain at the forefront of transportation innovation. Despite the impact of COVID-19, Lyft plans to continue investing in the future, both organically and through acquisitions of complementary businesses.</p>


#### Complex Queries

Let's test queries that requires multiple tools.

##### Query: What did Lyft do in R&D in 2022 vs 2020?

You can see that it used lyft_2020 and lyft_2022 tools with `FunctionCallingAgent` to answer the query.


```python
response = query_engine.query("What did Lyft do in R&D in 2022 vs 2020?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

[HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", Selecting query engine 3: This choice is most relevant as it spans multiple years from 2020 to 2022, which would allow for a comparison of Lyft's R&D activities in 2022 versus 2020.., Added user message to memory: What did Lyft do in R&D in 2022 vs 2020?, HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", === Calling Function ===, Calling function: lyft_2020_10k_form with args: {"input": "R&D expenses"}, HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", === Calling Function ===, Calling function: lyft_2022_10k_form with args: {"input": "R&D expenses"}, HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK", HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"]



<p style="font-size:20px">assistant: In 2020, Lyft's Research and Development expenses were $25,376 million, after accounting for a stock-based compensation benefit of $37,082 million and severance and other employee costs of $11,706 million. In 2022, these expenses decreased to $856,777 thousand, a decrease of $55.2 million, or 6%. This decrease was primarily due to a reduction in personnel-related costs and stock-based compensation, driven by reduced headcount following a transaction with Woven Planet in Q3 2021. There were also decreases in Level 5 costs, web hosting fees, autonomous vehicle research costs, and consulting and advisory costs. However, these decreases were partially offset by restructuring costs