# How to build a tool-using agent with LangChain

This notebook takes you through how to use LangChain to augment an OpenAI model with access to external tools. In particular, you'll be able to create LLM agents that use custom tools to answer user queries.

## What is Langchain?
[LangChain](https://python.langchain.com/en/latest/index.html) is a framework for developing applications powered by language models. Their framework enables you to build layered LLM-powered applications that are context-aware and able to interact dynamically with their environment as agents, leading to simplified code for you and a more dynamic user experience for your customers.

## Why do LLMs need to use Tools?
One of the most common challenges with LLMs is overcoming the lack of recency and specificity in their training data - answers can be out of date, and they are prone to hallucinations given the huge variety in their knowledge base. Tools are a great method of allowing an LLM to answer within a controlled context that draws on your existing knowledge bases and internal APIs - instead of trying to prompt engineer the LLM all the way to your intended answer, you allow it access to tools that it calls on dynamically for info, parses, and serves to customer. 

Providing LLMs access to tools can enable them to answer questions with context directly from search engines, APIs or your own databases. Instead of answering directly, an LLM with access to tools can perform intermediate steps to gather relevant information. Tools can also be used in combination. [For example](https://python.langchain.com/en/latest/modules/agents/agents/examples/mrkl_chat.html), a language model can be made to use a search tool to lookup quantitative information and a calculator to execute calculations.

## Notebook Sections

- **Setup:** Import packages and connect to a Pinecone vector database.
- **LLM Agent:** Build an agent that leverages a modified version of the [ReAct](https://react-lm.github.io/) framework to do chain-of-thought reasoning.
- **LLM Agent with History:** Provide the LLM with access to previous steps in the conversation.
- **Knowledge Base:** Create a knowledge base of "Stuff You Should Know" podcast episodes, to be accessed through a tool.
- **LLM Agent with Tools:** Extend the agent with access to multiple tools and test that it uses them to answer questions.

```python
%load_ext autoreload
%autoreload 2
```

# Setup

Import libraries and set up a connection to a [Pinecone](https://www.pinecone.io) vector database.

You can substitute Pinecone for any other vectorstore or database - there are a [selection](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html) that are supported by Langchain natively, while other connectors will need to be developed yourself.

```python
!pip install openai
!pip install pinecone-client
!pip install pandas
!pip install typing
!pip install tqdm
!pip install langchain
!pip install wget
```

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

# Langchain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
# Conversational memory
from langchain.memory import ConversationBufferWindowMemory
# Embeddings and vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Vectorstore Index
index_name = 'podcasts'
```

For acquiring an API key to connect with Pinecone, you can set up a [free account](https://app.pinecone.io/) and store it in the `api_key` variable below or in your environment variables under `PINECONE_API_KEY`

```python
api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"

# find environment next to your API key in the Pinecone console
env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"

pinecone.init(api_key=api_key, environment=env)
pinecone.whoami()
```

```python
pinecone.list_indexes()
```

Run this code block if you want to clear the index, or if the index doesn't exist yet

```
# Check whether the index with the same name already exists - if so, delete it
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)
    
# Creates new index
pinecone.create_index(name=index_name, dimension=1536)
index = pinecone.Index(index_name=index_name)

# Confirm our index was created
pinecone.list_indexes()
```

## LLM Agent

An [LLM agent](https://python.langchain.com/docs/modules/agents/) in Langchain has many configurable components, which are detailed in the Langchain documentation.

We'll employ a few of the core concepts to make an agent that talks in the way we want, can use tools to answer questions, and uses the appropriate language model to power the conversation.
- **Prompt Template:** The input template to control the LLM's behaviour and how it accepts inputs and produces outputs - this is the brain that drives your application ([docs](https://python.langchain.com/en/latest/modules/prompts/prompt_templates.html)).
- **Output Parser:** A method of parsing the output from the prompt. If the LLM produces output using certain headers, you can enable complex interactions where variables are generated by the LLM in their response and passed into the next step of the chain ([docs](https://python.langchain.com/en/latest/modules/prompts/output_parsers.html)).
- **LLM Chain:** A Chain brings together a prompt template with an LLM that will execute it - in this case we'll be using ```gpt-3.5-turbo``` but this framework can be used with OpenAI completions models, or other LLMs entirely ([docs](https://python.langchain.com/en/latest/modules/chains.html)).
- **Tool:** An external service that the LLM can use to retrieve information or execute commands should the user require it ([docs](https://python.langchain.com/en/latest/modules/agents/tools.html)).
- **Agent:** The glue that brings all of this together, an agent can call multiple LLM Chains, each with their own tools. Agents can be extended with your own logic to allow retries, error handling and any other methods you choose to add reliability to your application ([docs](https://python.langchain.com/en/latest/modules/agents.html)).

**NB:** Before using this cookbook with the Search tool you'll need to sign up on https://serpapi.com/ and generate an API key. Once you have it, store it in an environment variable named ```SERPAPI_API_KEY```

```python
# Initiate a Search tool - note you'll need to have set SERPAPI_API_KEY as an environment variable as per the above instructions
search = SerpAPIWrapper()

# Define a list of tools
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
```

```python
# Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
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
```

```python
# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
```

```python
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()
```

```python
# Initiate our LLM - default is 'gpt-3.5-turbo'
llm = ChatOpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    # We use "Observation" as our stop sequence so it will stop when it receives Tool output
    # If you change your prompt template you'll need to adjust this as well
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
```

```python
# Initiate the agent that will respond to our queries
# Set verbose=True to share the CoT reasoning the LLM goes through
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
```

```python
agent_executor.run("How many people live in canada as of 2023?")
```

```python
agent_executor.run("How many in 2022?")
```

## LLM Agent with History

Extend the LLM Agent with the ability to retain a [memory](https://python.langchain.com/en/latest/modules/agents/agents/custom_llm_agent.html#adding-memory) and use it as context as it continues the conversation.

We use a simple ```ConversationBufferWindowMemory``` for this example that keeps a rolling window of the last two conversation turns. LangChain has other [memory options](https://python.langchain.com/en/latest/modules/memory.html), with different tradeoffs suitable for different use cases.

```python
# Set up a prompt template which can interpolate the history
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
```

```python
prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # The history template includes "history" as an input variable so we can interpolate it into the prompt
    input_variables=["input", "intermediate_steps", "history"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
```

```python
# Initiate the memory with k=2 to keep the last two turns
# Provide the memory to the agent
memory = ConversationBufferWindowMemory(k=2)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
```

```python
agent_executor.run("How many people live in canada as of 2023?")
```

```python
agent_executor.run("how about in mexico?")
```

## Knowledge base

Create a custom vectorstore for the Agent to use as a tool to answer questions with. We'll store the results in [Pinecone](https://docs.pinecone.io/docs/quickstart), which is supported by LangChain ([Docs](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pinecone.html), [API reference](https://python.langchain.com/en/latest/reference/modules/vectorstore.html)). For help getting started with Pinecone or other vector databases, we have a [cookbook](https://github.com/openai/openai-cookbook/blob/colin/examples/vector_databases/Using_vector_databases_for_embeddings_search.ipynb) to help you get started.

You can check the LangChain documentation to see what other [vectorstores](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html) and [databases](https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html) are available.

For this example we'll use the transcripts of the Stuff You Should Know podcast, which was provided thanks to OSF DOI [10.17605/OSF.IO/VM9NT](https://doi.org/10.17605/OSF.IO/VM9NT)

```python
import wget

# Here is a URL to a zip archive containing the transcribed podcasts
# Note that this data has already been split into chunks and embeddings from OpenAI's `text-embedding-3-small` embedding model are included
content_url = 'https://cdn.openai.com/API/examples/data/sysk_podcast_transcripts_embedded.json.zip'

# Download the file (it is ~541 MB so this will take some time)
wget.download(content_url)
```

```python
# Load podcasts
with zipfile.ZipFile("sysk_podcast_transcripts_embedded.json.zip","r") as zip_ref:
    zip_ref.extractall("./data")
f = open('./data/sysk_podcast_transcripts_embedded.json')
processed_podcasts = json.load(f)
```

```python
# Have a look at the contents
pd.DataFrame(processed_podcasts).head()
```

```python
# Add the text embeddings to Pinecone

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(processed_podcasts), batch_size)):
    # find end of batch
    i_end = min(len(processed_podcasts), i+batch_size)
    meta_batch = processed_podcasts[i:i_end]
    # get ids
    ids_batch = [x['cleaned_id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text_chunk'] for x in meta_batch]
    # add embeddings
    embeds = [x['embedding'] for x in meta_batch]
    # cleanup metadata
    meta_batch = [{
        'filename': x['filename'],
        'title': x['title'],
        'text_chunk': x['text_chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
```

```python
# Configuring the embeddings to be used by our retriever to be OpenAI Embeddings, matching our embedded corpus
embeddings = OpenAIEmbeddings()


# Loads a docsearch object from an existing Pinecone index so we can retrieve from it
docsearch = Pinecone.from_existing_index(index_name,embeddings,text_key='text_chunk')
```

```python
retriever = docsearch.as_retriever()
