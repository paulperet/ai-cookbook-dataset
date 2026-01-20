# RAG Pipeline with Ollama, Mistral and LlamaIndex

In this notebook, we will demonstrate how to build a RAG pipeline using Ollama, Mistral models, and LlamaIndex. The following topics will be covered:

1.	Integrating Mistral with Ollama and LlamaIndex.
2.	Implementing RAG with Ollama and LlamaIndex using the Mistral model.
3.	Routing queries with RouterQueryEngine.
4.	Handling complex queries with SubQuestionQueryEngine.

Before running this notebook, you need to set up Ollama. Please follow the instructions [here](https://ollama.com/library/mistral:instruct).


```python
import nest_asyncio

nest_asyncio.apply()

from IPython.display import display, HTML
```

## Setup LLM


```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="mistral:instruct", request_timeout=60.0)

```

### Querying


```python

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is the capital city of France?"),
]
response = llm.chat(messages)
```


```python
display(HTML(f'<p style="font-size:20px">{response}</p>'))
```

assistant:  The capital city of France is Paris. It is located in the north-central part of the country and is one of the most populous cities in Europe. Paris is famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. The city is also known for its rich history, art, culture, fashion, and cuisine.

## Setup Embedding Model


```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

    /Users/ravithejad/Desktop/llamaindex/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    /Users/ravithejad/Desktop/llamaindex/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(



```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
```

## Download Data

We will use Uber and Lyft 10K SEC filings for the demostration.


```python
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O './lyft_2021.pdf'
```

    zsh:1: command not found: wget
    zsh:1: command not found: wget


## Load Data


```python
from llama_index.core import SimpleDirectoryReader

uber_docs = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()
lyft_docs = SimpleDirectoryReader(input_files=["./lyft_2021.pdf"]).load_data()
```

## Create Index and Query Engines


```python
from llama_index.core import VectorStoreIndex
from llama_index.core import SummaryIndex

uber_vector_index = VectorStoreIndex.from_documents(uber_docs)
uber_vector_query_engine = uber_vector_index.as_query_engine(similarity_top_k=2)

lyft_vector_index = VectorStoreIndex.from_documents(lyft_docs)
lyft_vector_query_engine = lyft_vector_index.as_query_engine(similarity_top_k=2)

```

### Querying


```python
response = uber_vector_query_engine.query("What is the revenue of uber in 2021 in millions?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

 The revenue of Uber in 2021 was $17,455 million.



```python
response = lyft_vector_query_engine.query("What is the revenue of lyft in 2021 in millions?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

 In the provided context, it can be found that the revenue for Lyft in the year ended December 31, 2021 was approximately $3,208 million (or $3,208,323 thousand).

## RouterQueryEngine

We will utilize the `RouterQueryEngine` to direct user queries to the appropriate index based on the query related to either Uber/ Lyft.

### Create QueryEngine tools


```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_vector_query_engine,
        metadata=ToolMetadata(
            name="vector_lyft_10k",
            description="Provides information about Lyft financials for year 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_vector_query_engine,
        metadata=ToolMetadata(
            name="vector_uber_10k",
            description="Provides information about Uber financials for year 2021",
        ),
    ),
]
```

### Create `RouterQueryEnine`


```python

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose = True
)
```

### Querying


```python
response = query_engine.query("What are the investments made by Uber?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    Selecting query engine 1: The provided choices are summaries of financial reports for the year 2021. Investments made by a company are not typically included in a financial report. However, the financial report may provide information about investments through capital expenditures or acquisition costs. As such, to find out about Uber's investments, one would need to look at a separate report or section that discusses these topics..


 Uber invests in a variety of financial instruments, as evidenced by the references to investments and equity method investments in the provided context. However, specific details about the nature or type of these investments are not directly disclosed in the information you've given. To determine the exact nature of these investments, one would need to review more detailed sections of the financial statements, such as Note 8 - Investments.



```python
response = query_engine.query("What are the investments made by the Lyft in 2021?")
```

    Selecting query engine 0: The given choices do not provide information about investments made by Lyft in 2021. They only provide information about the financials of Lyft and Uber for the year 2021..



```python
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

 The context provided indicates that Lyft invested in various areas in 2021. These include developing and launching new offerings and platform features, expanding in existing and new markets, continued investment in their platform and customer engagement, efforts to mitigate the impact of the COVID-19 pandemic, expansion of asset-intensive offerings such as Light Vehicles, Flexdrive, Lyft Rentals, Lyft Auto Care, and Driver Hubs, driver-centric service centers, and community spaces. They also expanded environmental programs, specifically their commitment to 100% EVs on their platform by the end of 2030. These offerings and programs require significant capital investments and recurring costs.

## SubQuestionQueryEngine

We will explore how the `SubQuestionQueryEngine` can be leveraged to tackle complex queries by generating and addressing sub-queries.

### Create `SubQuestionQueryEngine`


```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools,
                                                                 verbose=True)
```

### Querying


```python
response = sub_question_query_engine.query("Compare the revenues of Uber and Lyft in 2021?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    Generated 2 sub questions.
    [vector_uber_10k] Q: What is the revenue of Uber for year 2021
    [vector_lyft_10k] Q: What is the revenue of Lyft for year 2021
    [vector_lyft_10k] A:  In the provided context, the revenue for Lyft in the year 2021 is $3,208,323 thousand. This can be found on page 79 (file_path: lyft_2021.pdf).
    [vector_uber_10k] A:  The revenue for Uber in the year 2021, as per the provided context, was $17,455 million.


 In the year 2021, the revenue for Uber was $17,455 million and for Lyft it was $3,208,323 thousand. To compare, you can convert both revenues to the same units. In this case, since Uber's revenue is in millions, we can convert Lyft's revenue by dividing it by 1,000. Therefore, the comparable revenue for Lyft would be $3,208,323 / 1,000 = $3,208.323 million. So, Uber had a higher revenue than Lyft in 2021.



```python
response = sub_question_query_engine.query("What are the investments made by Uber and Lyft in 2021?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    Generated 2 sub questions.
    [vector_uber_10k] Q: What investments were made by Uber in year 2021
    [vector_lyft_10k] Q: What investments were made by Lyft in year 2021
    [vector_uber_10k] A:  In year 2021, Uber made purchases of non-marketable equity securities for 982 million dollars as per the provided cash flow statement. However, it is important to note that the information does not specify the details or names of the specific investments made.
    [vector_lyft_10k] A:  Based on the provided context, it appears that in the year 2021, Lyft made significant investments in several areas. These include developing and launching new offerings and platform features, expanding in existing and new markets, investing in their platform and customer engagement, investing in environmental programs such as their commitment to 100% EVs on their platform by the end of 2030, and expanding support services for drivers through Driver Hubs, Driver Centers, Mobile Services, Lyft AutoCare, and the Express Drive vehicle rental program. Additionally, they have invested in asset-intensive offerings like their network of Light Vehicles, Flexdrive, Lyft Rentals, and Lyft Auto Care. Furthermore, they have incurred and will continue to incur costs associated with Proposition 22 in California and the COVID-19 pandemic.


 In the year 2021, Uber made purchases of non-marketable equity securities for 982 million dollars as previously mentioned. On the other hand, Lyft invested in various areas such as developing and launching new offerings and platform features, expanding in existing and new markets, investing in their platform and customer engagement, environmental programs like a commitment to 100% EVs on their platform by the end of 2030, and support services for drivers through Driver Hubs, Driver Centers, Mobile Services, Lyft AutoCare, Express Drive vehicle rental program. Furthermore, Lyft has invested in asset-intensive offerings like their network of Light Vehicles, Flexdrive, Lyft Rentals, and Lyft Auto Care. Additionally, they have incurred and will continue to incur costs associated with Proposition 22 in California and the COVID-19 pandemic.