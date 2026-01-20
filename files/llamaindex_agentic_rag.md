# Using Mistral AI with LlamaIndex

In this notebook we're going to show how you can use LlamaIndex with the Mistral API to perform complex queries over multiple documents including answering questions that require multiple documents simultaneously. We'll do this using a ReAct agent, an autonomous LLM-powered agent capable of using tools.

First we install our dependencies. We need LlamaIndex, Mistral, and a PDF parser for later.

```python
!pip install llama-index-core 
!pip install llama-index-embeddings-mistralai
!pip install llama-index-llms-mistralai
!pip install llama-index-readers-file
!pip install mistralai pypdf
```

Now we set up our connection to Mistral. We need two things:
1. An LLM, to answer questions
2. An embedding model, to convert our data into vectors for retrieval by our index.
Luckily, Mistral provides both!

Once we have them, we put them into a ServiceContext, an object LlamaIndex uses to pass configuration around.

```python
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings

api_key = ""
llm = MistralAI(api_key=api_key,model="mistral-large-latest")
embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

Settings.llm = llm
Settings.embed_model = embed_model
```

Now let's download our dataset, 3 very large PDFs containing Lyft's annual reports from 2020-2022.

```python
!wget "https://www.dropbox.com/scl/fi/ywc29qvt66s8i97h1taci/lyft-10k-2020.pdf?rlkey=d7bru2jno7398imeirn09fey5&dl=0" -q -O ./lyft_10k_2020.pdf
!wget "https://www.dropbox.com/scl/fi/lpmmki7a9a14s1l5ef7ep/lyft-10k-2021.pdf?rlkey=ud5cwlfotrii6r5jjag1o3hvm&dl=0" -q -O ./lyft_10k_2021.pdf
!wget "https://www.dropbox.com/scl/fi/iffbbnbw9h7shqnnot5es/lyft-10k-2022.pdf?rlkey=grkdgxcrib60oegtp4jn8hpl8&dl=0" -q -O ./lyft_10k_2022.pdf
```

Now we have our data, we're going to do three things:
1. Load the PDF data into memory. It will be parsed into text as we do this. That's the `load_data()` line.
2. Index the data. This will create a vector representation of each document. That's the `from_documents()` line. It stores the vectors in memory.
3. Set up a query engine to retrieve information from the vector store and pass it to the LLM. That's the `as_query_engine()` line.

We're going to do this once for each of the three documents. If we had more than 3 we would do this programmatically with a loop, but this keeps the code very simple if a little repetitive. We've included a query to one of the indexes at the end as a test.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

lyft_2020_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2020.pdf"]).load_data()
lyft_2020_index = VectorStoreIndex.from_documents(lyft_2020_docs)
lyft_2020_engine = lyft_2020_index.as_query_engine()

lyft_2021_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2021.pdf"]).load_data()
lyft_2021_index = VectorStoreIndex.from_documents(lyft_2021_docs)
lyft_2021_engine = lyft_2021_index.as_query_engine()

lyft_2022_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2022.pdf"]).load_data()
lyft_2022_index = VectorStoreIndex.from_documents(lyft_2022_docs)
lyft_2022_engine = lyft_2022_index.as_query_engine()

response = lyft_2022_engine.query("What was Lyft's profit in 2022?")
print(response)
```

    Lyft did not make a profit in 2022. Instead, they incurred a net loss of $1,584,511.

Success! The 2022 index knows facts about 2022. We're almost ready to create our agent. Before we do, let's set up an array of tools for our agent to use. This turns each of the query engines we set up above into a tool, and indicates what each engine is best at answering questions about. The LLM will read these descriptions when deciding what tool to use.

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_2020_engine,
        metadata=ToolMetadata(
            name="lyft_2020_10k_form",
            description="Annual report of Lyft's financial activities in 2020",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2021_engine,
        metadata=ToolMetadata(
            name="lyft_2021_10k_form",
            description="Annual report of Lyft's financial activities in 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2022_engine,
        metadata=ToolMetadata(
            name="lyft_2022_10k_form",
            description="Annual report of Lyft's financial activities in 2022",
        ),
    ),
]

```

Now we create our agent from the tools we've set up and we can ask it complicated questions. It will reason through the process step by step, creating simpler questions, and use different tools to answer them. Then it'll take the information it gathers from each tool and combine it into a single answer to the more complex question.

```python
from llama_index.core.agent import ReActAgent

lyft_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
response = lyft_agent.chat("What are the risk factors in 2022?")
print(response)
```

    [Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
    Action: lyft_2022_10k_form
    Action Input: {'input': 'risk factors'}, ..., Answer: The risk factors for the company in 2022 include natural disasters such as earthquakes, fires, hurricanes, tornadoes, floods, or significant power outages, which could disrupt operations, mobile networks, the Internet, or the operations of third-party technology providers. The impact of climate change may increase these risks. Public health crises such as the COVID-19 pandemic, other epidemics, political crises like terrorist attacks, war, and other political or social instability, and other geopolitical developments, could also adversely affect operations or the economy as a whole. The company has offices and employees in regions like Belarus and Ukraine that have been and may continue to be adversely affected by current wars in the region, including displacement of employees. The company's limited operating history and evolving business make it difficult to evaluate future prospects and the risks and challenges that may be encountered. The company regularly expands its platform features, offerings, and services, and changes its pricing methodologies. The company's evolving business, industry, and markets make it difficult to evaluate future prospects and the risks and challenges that may be encountered. The company's business operations are also subject to numerous risks, factors, and uncertainties, including those outside of its control. These include general macroeconomic conditions, competition in the industry, the ability to attract and retain qualified drivers and riders, changes to pricing practices, and the ability to manage growth. The company's reliance on third parties, such as Amazon Web Services, vehicle rental partners, payment processors, and other service providers, also poses a risk. The development of new offerings on the platform and the management of the complexities of such expansion, as well as the ability to offer high-quality user support and deal with fraud, are additional risk factors. If the company fails to address these risks and difficulties, its business, financial condition, and results of operations could be adversely affected.]

```python
from llama_index.core.agent import ReActAgent

lyft_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
response = lyft_agent.chat("What is Lyft's profit in 2022 vs 2020? Generate only one step at a time. Use existing tools.")
print(response)
```

    [Thought: The current language of the user is: English. I need to use a tool to help me answer the question. First, I will find out Lyft's profit in 2020.
    Action: lyft_2020_10k_form
    Action Input: {'input': 'profit'}, ..., Answer: Lyft did not achieve profitability in 2022. Instead, the company reported a gross profit of $1,659.4 million and a Contribution of $1,729.8 million.]

Cool! As you can see it got the 2022 profit from the 2022 10-K form and the 2020 data from the 2020 report. It took both those answers and combined them into the difference we asked for. Let's try another question, this time asking about textual answers rather than numbers:

```python
from llama_index.core.agent import ReActAgent

lyft_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
response = lyft_agent.chat("What did Lyft do in R&D in 2022 versus 2021? Generate only one step at a time. Use existing tools.")
print(response)
```

    [Thought: The current language of the user is: English. I need to use a tool to help me answer the question. I will first retrieve the information about Lyft's R&D activities in 2022.
    Action: lyft_2022_10k_form
    Action Input: {'input': 'R&D'}, ..., Answer: In 2022, Lyft's R&D expenses decreased by 6% compared to the prior year, primarily due to a reduction in personnel-related costs and stock-based compensation, which were driven by reduced headcount following a transaction with Woven Planet in the third quarter of 2021. There were also reductions in Level 5 costs, web hosting fees, and autonomous vehicle research costs. However, these decreases were offset by restructuring costs related to an event in the fourth quarter of 2022, which included impairment costs of operating lease right-of-use assets, severance and benefits costs, and stock-based compensation. In 2021, Lyft's R&D expenses primarily consisted of personnel-related compensation costs and facilities costs. These expenses also included costs related to autonomous vehicle technology initiatives. The company expenses these costs as they are incurred. There was a transaction completed with Woven Planet, a subsidiary of Toyota Motor Corporation, on July 13, 2021, which involved the divestiture of certain assets related to the company's self-driving vehicle division. As a result, certain costs related to the prior initiative to develop self-driving systems were eliminated beginning in the third quarter of 2021.]

Great! It correctly itemized the risks, noticed the differences, and summarized them.

You can try this on any number of documents with any number of query engines to answer really complex questions. You can even have the query engines themselves be agents.