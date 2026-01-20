# Build a ReAct Agents with Mistral AI and LlamaIndex

This notebook shows you how to use `ReAct` Agent and `FunctionCalling` Agent over defined tools and RAG pipeline with MistralAI LLM.

### Installation

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

```python
!pip install llama-index
!pip install llama-index-llms-mistralai
!pip install llama-index-embeddings-mistralai
```

### Setup API Key

```python
import os
os.environ['MISTRAL_API_KEY'] = 'YOUR MISTRAL API KEY'
```

```python
import json
from typing import Sequence, List

from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

import nest_asyncio

nest_asyncio.apply()
```

Let's define some very simple calculator tools for our agent.

```python
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
```

```python
def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)
```

Make sure your MISTRAL_API_KEY is set. Otherwise explicitly specify the `api_key` parameter.

```python
llm = MistralAI(model="mistral-large-latest")
```

### With FunctionCalling Agent

Here we initialize a simple `FunctionCalling` agent with calculator functions.

```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = AgentRunner(agent_worker)
```

#### Chat

```python
response = agent.chat("What is (121 + 2) * 5?")
print(str(response))
```

Added user message to memory: What is (121 + 2) * 5?
=== Calling Function ===
Calling function: add with args: {"a": 121, "b": 2}
=== Calling Function ===
Calling function: multiply with args: {"a": 123, "b": 5}
assistant: The result of (121 + 2) * 5 is 615.

```python
# inspect sources
print(response.sources)
```

[ToolOutput(content='123', tool_name='add', raw_input={'args': (), 'kwargs': {'a': 121, 'b': 2}}, raw_output=123), ToolOutput(content='615', tool_name='multiply', raw_input={'args': (), 'kwargs': {'a': 123, 'b': 5}}, raw_output=615)]

#### Async Chat

Also let's re-enable parallel function calling so that we can call two `multiply` operations simultaneously.

```python
# enable parallel function calling
agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = AgentRunner(agent_worker)
response = await agent.achat("What is (121 * 3) + (5 * 8)?")
print(str(response))
```

Added user message to memory: What is (121 * 3) + (5 * 8)?
=== Calling Function ===
Calling function: multiply with args: {"a": 121, "b": 3}
=== Calling Function ===
Calling function: multiply with args: {"a": 5, "b": 8}
=== Calling Function ===
Calling function: add with args: {"a": 363, "b": 40}
assistant: The result of (121 * 3) + (5 * 8) is 403.

### With ReAct Agent

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + (5 * 8)?")
print(str(response))
```

[Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: multiply
Action Input: {"a": 121, "b": 3}

Observation: 363

Thought: I need to use another tool to complete the calculation.
Action: multiply
Action Input: {"a": 5, "b": 8}

Observation: 40

Thought: I need to use one more tool to complete the calculation.
Action: add
Action Input: {"a": 363, "b": 40}

Observation: 403

Thought: I can answer without using any more tools. I'll use the user's language to answer.
Answer: The result of (121 * 3) + (5 * 8) is 403., ..., The result of (121 * 3) + (5 * 8) is 403.]

### Agent over RAG Pipeline

Build a Mistral FunctionCalling agent over a simple 10K document. We use both Mistral embeddings and mistral-medium to construct the RAG pipeline, and pass it to the Mistral agent as a tool.

```python
!mkdir -p 'data/10k/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
```

[--2024-04-03 19:01:37--  https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf
Resolving raw.githubusercontent.com (raw.githubusercontent.com)..., ..., 185.199.111.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1880483 (1.8M) [application/octet-stream]
Saving to: ‘data/10k/uber_2021.pdf’

data/10k/uber_2021. 100%[===================>]   1.79M  --.-KB/s    in 0.04s   

2024-04-03 19:01:38 (40.3 MB/s) - ‘data/10k/uber_2021.pdf’ saved [1880483/1880483], ..., data/10k/uber_2021. 100%[===================>]   1.79M  --.-KB/s    in 0.04s   

2024-04-03 19:01:38 (40.3 MB/s) - ‘data/10k/uber_2021.pdf’ saved [1880483/1880483]]

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

embed_model = MistralAIEmbedding()
query_llm = MistralAI(model="mistral-medium")

# load data
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()
# build index
uber_index = VectorStoreIndex.from_documents(
    uber_docs, embed_model=embed_model
)
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)
query_engine_tool = QueryEngineTool(
    query_engine=uber_engine,
    metadata=ToolMetadata(
        name="uber_10k",
        description=(
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
        ),
    ),
)
```

### With FunctionCalling Agent

```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    [query_engine_tool], llm=llm, verbose=True
)
agent = AgentRunner(agent_worker)
```

```python
response = agent.chat(
    "What are the risk factors for Uber in 2021?"
)
print(str(response))
```

[Thought: The current language of the user is: English. I need to use the uber_10k tool to help me answer the question.
Action: uber_10k
Action Input: {'input': 'What are the risk factors for Uber in 2021?'}
Observation: Uber faces several risk factors in 2021, including:

1. Failure to offer or develop autonomous vehicle technologies, which could result in inferior performance or safety concerns compared to competitors.
2. Dependence on retaining and attracting high-quality personnel, with attrition or unsuccessful succession planning potentially harming the business.
3. Security or data privacy breaches, unauthorized access, or destruction of proprietary, employee, or user data.
4. Cyberattacks, such as malware, ransomware, viruses, spamming, and phishing attacks, which could harm the company's reputation and operations.
5. Climate change risks, including physical and transitional risks, which may require significant investment of resources and management time.
6. Reliance on third parties to maintain open marketplaces for distributing Uber's platform and software, which could be disrupted and negatively impact the business.
7. The need for additional capital to support business growth, which may not be available on reasonable terms or at all.
8. Challenges in identifying, acquiring, and integrating suitable businesses, which could harm operating results and prospects.
9. Legal and regulatory risks, including potential blocks or limitations in providing or operating products and offerings in certain jurisdictions.
10. Extensive government regulation and oversight related to payment and financial services.
11. Risks related to data collection, use, transfer, disclosure, and processing, which could result in investigations, fines, legislative and regulatory action, and negative press.
12. Intellectual property risks, including potential claims of misappropriation by third parties.
13. Market price volatility for Uber's common stock, which could decline steeply or suddenly regardless of operating performance, potentially resulting in significant losses for investors.
14. Economic risks related to the COVID-19 pandemic, which has adversely impacted and could continue to adversely impact Uber's business, financial condition, and results of operations.
15. Decline in the number of drivers, consumers, merchants, shippers, or carriers using the platform, which would reduce the value of the network and harm future operating results.

Thought: The current language of the user is: English. I have already provided a detailed answer about the risk factors for Uber in 2021 using the uber_10k tool.
Answer: Uber faces several risk factors in 2021, including:

1. Failure to offer or develop autonomous vehicle technologies, which could result in inferior performance or safety concerns compared to competitors.
2. Dependence on retaining and attracting high-quality personnel, with attrition or unsuccessful succession planning potentially harming the business.
3. Security or data privacy breaches, unauthorized access, or destruction of proprietary, employee, or user data.
4. Cyberattacks, such as malware, ransomware, viruses, spamming, and phishing attacks, which could harm the company's reputation and operations.
5. Climate change risks, including physical and transitional risks, which may require significant investment of resources and management time.
6. Reliance on third parties to maintain open marketplaces for distributing Uber's platform and software, which could be disrupted and negatively impact the business.
7. The need for additional capital to support business growth, which may not be available on reasonable terms or at all.
8. Challenges in identifying, acquiring, and integrating suitable businesses, which could harm operating results and prospects.
9. Legal and regulatory risks, including potential blocks or limitations in providing or operating products and offerings in certain jurisdictions.
10. Extensive government regulation and oversight related to payment and financial services.
11. Risks related to data collection, use, transfer, disclosure, and processing, which could result in investigations, fines, legislative and regulatory action, and negative press.
12. Intellectual property risks, including potential claims of misappropriation by third parties.
13. Market price volatility for Uber's common stock, which could decline steeply or suddenly regardless of operating performance, potentially resulting in significant losses for investors.
14. Economic risks related to the COVID-19 pandemic, which has adversely impacted and could continue to adversely impact Uber's business, financial condition, and results of operations.
15. Decline in the number of drivers, consumers, merchants, shippers, or carriers using, ..., Uber faces several risk factors in 2021, including:

1. Failure to offer or develop autonomous vehicle technologies, which could result in inferior performance or safety concerns compared to competitors.
2. Dependence on retaining and attracting high-quality personnel, with attrition or unsuccessful succession planning potentially harming the business.
3. Security or data privacy breaches, unauthorized access, or destruction of proprietary, employee, or user data.
4. Cyberattacks, such as malware, ransomware, viruses, spamming, and phishing attacks, which could harm the company's reputation and operations.
5. Climate change risks, including physical and transitional risks, which may require significant investment of resources and management time.
6. Reliance on third parties to maintain open marketplaces for distributing Uber's platform and software, which could be disrupted and negatively impact the business.
7. The need for additional capital to support business growth, which may not be available on reasonable terms or at all.
8. Challenges in identifying, acquiring, and integrating suitable businesses, which could harm operating results and prospects.
9. Legal and regulatory risks, including potential blocks or limitations in providing or operating products and offerings in certain jurisdictions.
10. Extensive government regulation and oversight related to payment and financial services.
11. Risks related to data collection, use, transfer, disclosure, and processing, which could result in investigations, fines, legislative and regulatory action, and negative press.
12. Intellectual property risks, including potential claims of misappropriation by third parties.
13. Market price volatility for Uber's common stock, which could decline steeply or suddenly regardless of operating performance, potentially resulting in significant losses for investors.
14. Economic risks related to the COVID-19 pandemic, which has adversely impacted and could continue to adversely impact Uber's business, financial condition, and results of operations.
15. Decline in the number of drivers, consumers, merchants, shippers, or carriers using]

### With ReAct Agent

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools([query_engine_tool], llm=llm, verbose=True)

response = agent.chat("What are the risk factors for Uber in 2021?")
print(str(response))
```

[Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: uber_10k
Action Input: {'input': 'What are the risk factors for Uber in 2021?'}
Observation: Uber faces several risk factors in 2021, including:

1. Autonomous vehicle technology: Uber may fail to offer autonomous vehicle technologies on its platform, or such technologies may not perform as expected, be inferior to competitors, or be perceived as less safe.
2. Personnel: Uber's business depends on retaining and attracting high-quality personnel, and continued attrition or unsuccessful succession planning could adversely affect its business.
3. Data privacy and security breaches: Uber may experience security or data privacy breaches, which could harm its reputation, business, and operating results.
4. Cyberattacks: Cyberattacks, including computer malware, ransomware, viruses, spamming, and phishing attacks, could harm Uber's reputation, business, and operating results.
5. Climate change risks: Uber is subject to climate change risks, including physical and transitional risks, and if it is unable to manage such risks, its business may be adversely impacted.
6. Third-party marketplaces: Uber relies on third parties maintaining open marketplaces to distribute its platform and provide software for its products and offerings. If such third parties interfere with the distribution of Uber's products or offerings or with its use of such software, its business would be adversely affected.
7. Capital requirements: Uber will require additional capital to support the growth of its business, and this capital might not be available on reasonable terms or at all.
8. Acquisitions: If Uber is unable to successfully identify, acquire, and integrate suitable businesses, its operating results and prospects could be harmed, and any businesses it acquires may not perform as expected or be effectively integrated.
9. Legal and regulatory risks: Uber's business is subject to numerous legal and regulatory risks that could have an adverse impact on its business and future prospects.
10. Payment and financial services regulation: Uber's business is subject to extensive government regulation and oversight relating to the provision of payment and financial services.
11. Data processing risks: Uber faces risks related to its collection, use, transfer, disclosure, and other processing of data, which could result in investig

Thought: The current language of the user is: English. I have already used the 'uber_10k' tool to provide the risk factors for Uber in 2021. I can now answer without using any more tools.
Answer: Uber faces several risk factors in 2021, including:

1. Autonomous vehicle technology: Uber may fail to offer autonomous vehicle technologies on its platform, or such technologies may not perform as expected, be inferior to competitors, or be perceived as less safe.
2. Personnel: Uber's business depends on retaining and attracting high-quality personnel, and continued attrition or unsuccessful succession planning could adversely affect its business.
3. Data privacy and security breaches: Uber may experience security or data privacy breaches, which could harm its reputation, business, and operating results.
4. Cyberattacks: Cyberattacks, including computer malware, ransomware, viruses, spamming, and phishing attacks, could harm Uber's reputation, business, and operating results.
5. Climate change risks: Uber is subject to climate change risks, including physical and transitional risks, and if it is unable to manage such risks, its business may be adversely impacted.
6. Third-party marketplaces: Uber relies on third parties maintaining open marketplaces to distribute its platform and provide software for its products and offerings. If such third parties interfere with the distribution of Uber's products or offerings or with its use of such software, its business would be adversely affected.
7. Capital requirements: Uber will require additional capital to support the growth of its business, and this capital might not be available on reasonable terms or at all.
8. Acquisitions: If Uber is unable to successfully identify, acquire, and integrate suitable businesses, its operating results and prospects could be harmed, and any businesses it acquires may not perform as expected or be effectively integrated.
9. Legal and regulatory risks: Uber's