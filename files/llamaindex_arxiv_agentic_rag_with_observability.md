# Building an LLM Agent to Find Relevant Research Papers from Arxiv

This notebook was created by Andrei Chernov ([Github](https://github.com/ChernovAndrey), [Linkedin](https://www.linkedin.com/in/andrei-chernov-58b157236/))
In this tutorial, we will create an LLM agent based on the **MistralAI** language model. The agent's primary purpose will be to find and summarize research papers from **Arxiv** that are relevant to the user's query. To build the agent, we will use the **LlamaIndex** framework.

## Tools Used by the Agent

The agent will utilize the following three tools:

1. **RAG Query Engine**
   This tool will store and retrieve recent papers from Arxiv, serving as a knowledge base for efficient and quick access to relevant information.

2. **Paper Fetch Tool**
   If the user specifies a topic that is not covered in the RAG Query Engine, this tool will fetch recent papers on the specified topic directly from Arxiv.

3. **PDF Download Tool**
   This tool allows the agent to download a research paper's PDF file locally using a link provided by Arxiv.

### First, let's install necessary libraries


```python
!pip install arxiv==2.1.3 llama_index==0.12.3 llama-index-llms-mistralai==0.3.0 llama-index-embeddings-mistralai==0.3.0 
!pip install arize-phoenix==7.2.0 arize-phoenix-evals==0.18.0 openinference-instrumentation-llama-index==3.0.2
```

    [Requirement already satisfied: arxiv==2.1.3, ..., openinference-instrumentation-llama-index==3.0.2]



```python
from getpass import getpass
import requests
import sys
import arxiv
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, PromptTemplate, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent

```

### Additionally, You Need to Provide Your API Key to Access Mistral Models

You can obtain an API key [here](https://console.mistral.ai/api-keys/).


```python
api_key= getpass("Type your API Key")
```


```python
llm = MistralAI(api_key=api_key, model='mistral-large-latest')
```

### To Build a RAG Query Engine, We Will Need an Embedding Model

For this tutorial, we will use the MistralAI embedding model.


```python
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)

```

### Now, We Will Download Recent Papers About Large Language Models from ArXiv

To keep this tutorial accessible with the free Mistral API version, we will download only the last 10 papers. Downloading more would exceed the limit later while building the RAG query engine. However, if you have a Mistral subscription, you can download additional papers.


```python
def fetch_arxiv_papers(title :str, papers_count: int):
    search_query = f'all:"{title}"'
    search = arxiv.Search(
        query=search_query,
        max_results=papers_count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    # Use the Client for searching
    client = arxiv.Client()
    
    # Execute the search
    search = client.results(search)

    for result in search:
        paper_info = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published,
                'journal_ref': result.journal_ref,
                'doi': result.doi,
                'primary_category': result.primary_category,
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'arxiv_url': result.entry_id
            }
        papers.append(paper_info)

    return papers
    
papers = fetch_arxiv_papers("Language Models", 10)
```


```python
[[p['title']] for p in papers]
```




    [['PRIMA: Multi-Image Vision-Language Models for Reasoning Segmentation'],
     ['OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving'],
     ['AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving'],
     ['MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark'],
     ['EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues'],
     ['LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation'],
     ['Data for Mathematical Copilots: Better Ways of Presenting Proofs for Machine Learning'],
     ['HPC-Coder-V2: Studying Code LLMs Across Low-Resource Parallel Languages'],
     ['Critical-Questions-of-Thought: Steering LLM reasoning with Argumentative Querying'],
     ['Rethinking Uncertainty Estimation in Natural Language Generation']]



### To Build a RAG Agent, We First Need to Index All Documents

This process creates a vector representation for each chunk of a document using the embedding model.


```python
def create_documents_from_papers(papers):
    documents = []
    for paper in papers:
        content = f"Title: {paper['title']}\n" \
                  f"Authors: {', '.join(paper['authors'])}\n" \
                  f"Summary: {paper['summary']}\n" \
                  f"Published: {paper['published']}\n" \
                  f"Journal Reference: {paper['journal_ref']}\n" \
                  f"DOI: {paper['doi']}\n" \
                  f"Primary Category: {paper['primary_category']}\n" \
                  f"Categories: {', '.join(paper['categories'])}\n" \
                  f"PDF URL: {paper['pdf_url']}\n" \
                  f"arXiv URL: {paper['arxiv_url']}\n"
        documents.append(Document(text=content))
    return documents



#Create documents for LlamaIndex
documents = create_documents_from_papers(papers)
```


```python
Settings.chunk_size = 1024
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
```

### Now, We Will Store the Index

Indexing a large number of texts can be time-consuming and costly since it requires making API calls to the embedding model. In real-world applications, it is better to store the index in a vector database to avoid reindexing. However, for simplicity, we will store the index locally in a directory in this tutorial, without using a vector database.


```python
index.storage_context.persist('index/')
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='index/')

#load index
index = load_index_from_storage(storage_context, embed_model=embed_model)
```

### We Are Ready to Build a RAG Query Engine for Our Agent

It is a good practice to provide a meaningful name and a clear description for each tool. This helps the agent select the most appropriate tool when needed.


```python
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="research_paper_query_engine_tool",
    description="A RAG engine with recent research papers.",
)
```

### Let's Take a Look at the Prompts the RAG Tool Uses to Answer a Query Based on Context

Note that there are two prompts. By default, LlamaIndex uses a refine prompt before returning an answer. You can find more information about the response modes [here](https://docs.llamaindex.ai/en/v0.10.34/module_guides/deploying/query_engine/response_modes/).


```python
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display
# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))
        
prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)
```


**Prompt Key**: response_synthesizer:text_qa_template**Text:** 


    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: 







**Prompt Key**: response_synthesizer:refine_template**Text:** 


    The original query is as follows: {query_str}
    We have provided an existing answer: {existing_answer}
    We have the opportunity to refine the existing answer (only if needed) with some more context below.
    ------------
    {context_msg}
    ------------
    Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
    Refined Answer: 






### Building two other tools is straightforward because they are simply Python functions.


```python
def download_pdf(pdf_url, output_file):
    """
    Downloads a PDF file from the given URL and saves it to the specified file.

    Args:
        pdf_url (str): The URL of the PDF file to download.
        output_file (str): The path and name of the file to save the PDF to.

    Returns:
        str: A message indicating success or the nature of an error.
    """
    try:
        # Send a GET request to the PDF URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Write the content of the PDF to the output file
        with open(output_file, "wb") as file:
            file.write(response.content)

        return f"PDF downloaded successfully and saved as '{output_file}'."

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
```


```python
download_pdf_tool = FunctionTool.from_defaults(
    download_pdf,
    name='download_pdf_file_tool',
    description='python function, which downloads a pdf file by link'
)
fetch_arxiv_tool = FunctionTool.from_defaults(
    fetch_arxiv_papers,
    name='fetch_from_arxiv',
    description='download the {max_results} recent papers regarding the topic {title} from arxiv' 
)

```


```python
# building an ReAct Agent with the three tools.
agent = ReActAgent.from_tools([download_pdf_tool, rag_tool, fetch_arxiv_tool], llm=llm, verbose=True)
```

### Let's Chat with Our Agent

We built a ReAct agent, which operates in two main stages:

1. **Reasoning**: Upon receiving a query, the agent evaluates whether it has enough information to answer directly or if it needs to use a tool.
2. **Acting**: If the agent decides to use a tool, it executes the tool and then returns to the Reasoning stage to determine whether it can now answer the query or if further tool usage is necessary.


```python
# create a prompt template to chat with an agent
q_template = (
    "I am interested in {topic}. \n"
    "Find papers in your knowledge database related to this topic; use the following template to query research_paper_query_engine_tool tool: 'Provide title, summary, authors and link to download for papers related to {topic}'. If there are not, could you fetch the recent one from arXiv? \n"
)
```


```python
answer = agent.chat(q_template.format(topic="Audio Models"))
```

    [Running step 8b49615f-d6e7-4b51-bfcf-a900909fb957, ..., Running step 3bb09466-9000-473e-8a87-32d0236e93e0]
    [Thought: The current language of the user is: English. I need to use a tool to help me answer the question., ..., Thought: I can answer without using any more tools. I'll use the user's language to answer]


```python
Markdown(answer.response)
```




I'm afraid there are no papers related to Audio Models in the provided information.

However, I found some recent papers from arXiv:

1. **Title**: Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes
   - **Authors**: Kuiyuan Zhang, Zhongyun Hua, Rushi Lan, Yushu Zhang, Yifang Guo
   - **Summary**: This paper introduces a new mechanism for detecting speech deepfakes by identifying inconsistencies in phoneme-level speech features. The authors propose an adaptive phoneme pooling technique and a graph attention network to model temporal dependencies, demonstrating superior performance over existing detection methods.
   - **Link to download**: http://arxiv.org/pdf/2412.12619v1

2. **Title**: Hidden Echoes Survive Training in Audio To Audio Generative Instrument Models
   - **Authors**: Christopher J. Tralie, Matt Amery, Benjamin Douglas, Ian Utz
   - **Summary**: This paper explores how generative audio models draw on their training data to synthesize new examples. The authors show that imperceptible echoes hidden in the training data are reproduced in the outputs of various audio-to-audio architectures, demonstrating the robustness of this watermarking technique.
   - **Link to download**: http://arxiv.org/pdf/2412.10649v1

3. **Title**: AdvWave: Stealthy Adversarial Jailbreak Attack against Large Audio-Language Models
   - **Authors**: Mintong Kang, Chejian Xu, Bo Li
   - **Summary**: This paper introduces AdvWave, a framework for jailbreaking large audio-language models. The authors propose a dual-phase optimization method and an adaptive adversarial target search algorithm to overcome the challenges of gradient shattering and behavioral variability in LALMs.
   - **Link to download**: http://arxiv.org/pdf/2412.08608v



### The agent chose to use the RAG tool, found the relevant papers, and summarized them for us.  
### Since the agent retains the chat history, we can request to download the papers without mentioning them explicitly.


```python
answer = agent.chat("Download the papers, which you mentioned above")
```

    [Running step bc66b185-dbd9-440f-8bae-0729202e7d10, ..., Running step fb6783c5-3cfe-4640-ab85-5ca3be89e461]
    [Thought: I need to use a tool to help me answer the question., ..., Thought: I can answer without using any more tools. I'll use the user's language to answer]


```python
Markdown(answer.response)
```




All papers were downloaded successfully.



### Let's see what happens if we ask about a topic that is not available in the RAG.


```python
answer = agent.chat(q_template.format(topic="Gaussian process"))
```

    [Running step f0e45bd7-3a3b-4d4f-a114-299abb01c446, ..., Running step a70b797b-b529-45f3-8e42-5eb965726a13]
    [Thought: I need to use a tool to help me answer the question., ..., Thought: (Implicit) I can answer without any more tools!]


```python
Markdown(answer.response)
```




I found some recent papers from arXiv:

1. **Title**: Co-optimization of Vehicle Dynamics and Powertrain Management for Connected and Automated Electric Vehicles
   - **Authors**: Zongtan Li, Yunli Shao
   - **Summary**: This paper studies the co-optimization of vehicle speed and powertrain management for energy savings in a dual-motor electric vehicle. The authors developed control-oriented vehicle dynamics and electric powertrain models to transform the problem into an optimal control problem designed for real-time computation. Simulation results demonstrated a significant reduction in power consumption under ideal predicted traffic conditions.
   - **Link to download**: http://arxiv.org/pdf/2412.14984v1

2. **Title**: Comparing noisy neural population dynamics using optimal transport distances
   - **Authors**: Amin Nejatbakhsh, Victor Geadah, Alex H. Williams, David Lipshutz
   - **Summary**: This paper proposes a metric for comparing the geometry of noisy neural trajectories, derived as an optimal transport distance between Gaussian processes. The metric is used to compare models of neural responses in different regions of the motor system and to compare the dynamics of latent diffusion models for text-to-image synthesis.
   - **Link to download**: http://arxiv.org/pdf/2412.14421v1

3. **Title**: Model-Agnostic Cosmological Inference with SDSS-IV eBOSS: Simultaneous Probing for Background and Perturbed Universe
   - **Authors**: Purba Mukherjee, Anjan A. Sen
   - **Summary**: This paper explores features in data from the SDSS-IV eBOSS survey as a combined probe for the background and perturbed Universe. The authors use the Multi-Task Gaussian Process framework to model the interdependencies of cosmological observables and obtain constraints on model parameters, suggesting a possible inconsistency in the ΛCDM model.
   - **Link to download**: http://arxiv.org/pdf/2412.13973v1



### As You Can See, the Agent Did Not Find the Papers in Storage and Fetched Them from ArXiv.

For a more detailed view of the agent's execution, check out your [Phoenix](https://app.phoenix.arize.com) dashboard.

# (Optional) Let's Trace and Evaluate the Agent

LlamaIndex has a built-in observability layer powered by Arize Phoenix. We can use this to trace the agent's execution and evaluate its performance.

If you don't have a Phoenix API key, you can get one [here](https://app.phoenix.arize.com/login/sign-up).


```python
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import os

PHOENIX_API_KEY = getpass("Type your Phoenix API Key")
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer = register(project_name="arxiv-agentic-rag")
LlamaIndexInstrumentor().instrument