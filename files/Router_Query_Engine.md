# RouterQuery Engine

In this notebook we will look into `RouterQueryEngine` to route the user queries to one of the available query engine tools. These tools can be different indices/ query engine on same documents/ different documents.

### Installation

```python
!pip install llama-index
!pip install llama-index-llms-anthropic
!pip install llama-index-embeddings-huggingface
```

### Set Logging

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

from IPython.display import HTML, display
```

### Set Claude API Key

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "YOUR Claude API KEY"
```

### Set LLM and Embedding model

We will use anthropic latest released `Claude-3 Opus` LLM.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
```

```python
llm = Anthropic(temperature=0.0, model="claude-opus-4-1")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
```

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

### Download Document

```
--2024-03-08 07:04:27--  https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.110.133, 185.199.108.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 75042 (73K) [text/plain]
Saving to: ‘data/paul_graham/paul_graham_essay.txt’

data/paul_graham/pa 100%[===================>]  73.28K  --.-KB/s    in 0.002s  

2024-03-08 07:04:27 (28.6 MB/s) - ‘data/paul_graham/paul_graham_essay.txt’ saved [75042/75042]
```

### Load Document

```python
# load documents
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data/paul_graham").load_data()
```

### Create Indices and Query Engines.

```python
from llama_index.core import SummaryIndex, VectorStoreIndex

# Summary Index for summarization questions
summary_index = SummaryIndex.from_documents(documents)

# Vector Index for answering specific context questions
vector_index = VectorStoreIndex.from_documents(documents)
```

```python
# Summary Index Query Engine
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

# Vector Index Query Engine
vector_query_engine = vector_index.as_query_engine()
```

### Create tools for summary and vector query engines.

```python
from llama_index.core.tools.query_engine import QueryEngineTool

# Summary Index tool
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to Paul Graham eassy on What I Worked On.",
)

# Vector Index tool
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)
```

### Create Router Query Engine

```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
```

```python
# Create Router Query Engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)
```

### Test Queries

```python
response = query_engine.query("What is the summary of the document?")
```

[HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"]

```python
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

The document is an autobiographical essay by Paul Graham, describing the major projects and events in his life from childhood through his 50s. Key points include:

- As a child, he was interested in programming and writing. He attended college intending to study philosophy but switched to AI.

- After grad school, he decided to pursue art and attended RISD and the Accademia in Florence. He supported himself doing freelance Lisp programming. 

- In 1995, he and Robert Morris started Viaweb, one of the first web application companies, which was acquired by Yahoo in 1998. This made Graham wealthy.

- After leaving Yahoo, he returned to painting for a time, then started publishing essays online and working on a new Lisp dialect called Arc. 

- In 2005, he co-founded Y Combinator, a new kind of startup investment firm, with Jessica Livingston, Robert Morris and Trevor Blackwell. He was very engaged in YC for many years as it pioneered a new model of startup funding.

- In 2013 he handed over the reins of YC to Sam Altman. After a period focused on painting, in 2015 he began developing a new Lisp dialect called Bel, which he worked on intensively for 4 years.

- The essay reflects on the winding path his career took, the accidental discoveries like the YC model, and the way his interests in art, writing and programming languages have intertwined over the decades. It emphasizes the value he found in pursuing ideas despite their unprestigious status.

```python
response = query_engine.query("What did Paul Graham do growing up?")
```

[HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"]

```python
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

According to the context provided, the two main things Paul Graham worked on outside of school before college were writing and programming. He wrote short stories, which he admits were awful, with hardly any plot and just characters with strong feelings. 

In 9th grade, when he was 13 or 14, he started trying to write programs on an IBM 1401 computer that his school district used. The language was an early version of Fortran, and programs had to be typed on punch cards. However, he was puzzled by the 1401 and couldn't figure out what to really do with it, since the only input was data on punched cards which he didn't have. His clearest memory is learning that programs could fail to terminate when one of his didn't, which was a social faux pas on a shared machine. But everything changed for him once microcomputers became available.