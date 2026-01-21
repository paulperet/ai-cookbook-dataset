# Building a Tool-Using Agent with LangChain

## Overview
This guide demonstrates how to use LangChain to create an LLM agent that can access external tools to answer user queries. You'll build an agent that leverages the ReAct framework for reasoning, maintains conversation history, and integrates a custom knowledge base.

## Prerequisites

Install the required Python packages:

```bash
pip install openai pinecone-client pandas typing tqdm langchain wget
```

## 1. Initial Setup

### Import Libraries and Configure Pinecone

```python
import datetime
import json
import openai
import os
import pandas as pd
import pinecone
import re
from tqdm.auto import tqdm
from typing import List, Union
import zipfile

# LangChain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Vectorstore configuration
index_name = 'podcasts'
```

### Initialize Pinecone Connection

Create a [free Pinecone account](https://app.pinecone.io/) and store your API key in the environment variable `PINECONE_API_KEY`.

```python
api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"

pinecone.init(api_key=api_key, environment=env)
pinecone.whoami()
```

### Create or Clear the Index

```python
# List existing indexes
pinecone.list_indexes()

# Create a new index (delete if it already exists)
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)
    
pinecone.create_index(name=index_name, dimension=1536)
index = pinecone.Index(index_name=index_name)

# Confirm index creation
pinecone.list_indexes()
```

## 2. Building a Basic LLM Agent

LangChain agents consist of several key components:
- **Prompt Template**: Controls LLM behavior and input/output formatting
- **Output Parser**: Parses LLM output to extract actions and variables
- **LLM Chain**: Combines prompt template with an LLM
- **Tool**: External services the LLM can use
- **Agent**: Orchestrates chains and tools with custom logic

### Step 1: Define Tools

First, create a search tool. You'll need a [SerpAPI](https://serpapi.com/) account and API key stored as `SERPAPI_API_KEY`.

```python
search = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
```

### Step 2: Create a Custom Prompt Template

This template implements the ReAct framework with a pirate-themed personality.

```python
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)
```

### Step 3: Implement an Output Parser

This parser extracts actions or final answers from the LLM's response.

```python
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()
```

### Step 4: Assemble the Agent

```python
llm = ChatOpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
```

### Step 5: Test the Agent

```python
agent_executor.run("How many people live in canada as of 2023?")
```

The agent will use the search tool and respond with a pirate-themed answer.

## 3. Adding Conversation Memory

Extend the agent to remember previous conversation turns using `ConversationBufferWindowMemory`.

### Step 1: Create a Memory-Enabled Prompt

```python
template_with_history = """You are SearchGPT, a professional search engine who provides informative answers to users. Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to give detailed, informative answers

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)
```

### Step 2: Reconfigure the Agent with Memory

```python
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

memory = ConversationBufferWindowMemory(k=2)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    memory=memory
)
```

### Step 3: Test the Conversational Agent

```python
agent_executor.run("How many people live in canada as of 2023?")
agent_executor.run("how about in mexico?")
```

The agent will now maintain context between questions.

## 4. Creating a Custom Knowledge Base Tool

Add a vector database tool that provides information from podcast transcripts.

### Step 1: Download and Prepare the Data

```python
import wget

# Download podcast transcripts (541 MB file)
content_url = 'https://cdn.openai.com/API/examples/data/sysk_podcast_transcripts_embedded.json.zip'
wget.download(content_url)

# Extract and load the data
with zipfile.ZipFile("sysk_podcast_transcripts_embedded.json.zip","r") as zip_ref:
    zip_ref.extractall("./data")
    
with open('./data/sysk_podcast_transcripts_embedded.json') as f:
    processed_podcasts = json.load(f)

# Preview the data structure
pd.DataFrame(processed_podcasts).head()
```

### Step 2: Populate the Vector Database

```python
batch_size = 100

for i in tqdm(range(0, len(processed_podcasts), batch_size)):
    i_end = min(len(processed_podcasts), i+batch_size)
    meta_batch = processed_podcasts[i:i_end]
    ids_batch = [x['cleaned_id'] for x in meta_batch]
    texts = [x['text_chunk'] for x in meta_batch]
    embeds = [x['embedding'] for x in meta_batch]
    
    meta_batch = [{
        'filename': x['filename'],
        'title': x['title'],
        'text_chunk': x['text_chunk'],
        'url': x['url']
    } for x in meta_batch]
    
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    index.upsert(vectors=to_upsert)
```

### Step 3: Configure the Retriever

```python
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name, embeddings, text_key='text_chunk')
retriever = docsearch.as_retriever()
```

## Next Steps

You now have a functional agent that can:
1. Use external tools (like search) to answer questions
2. Maintain conversation history
3. Access a custom knowledge base

To extend this agent further:
- Add the retriever as a new tool in the `tools` list
- Implement error handling and retry logic
- Experiment with different memory types for longer conversations
- Add more specialized tools for your use case

The modular nature of LangChain allows you to swap components easily—try different LLMs, vector databases, or prompt templates to optimize for your specific requirements.