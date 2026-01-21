# Building an LLM Agent to Find Relevant Research Papers from Arxiv

**Author:** Andrei Chernov ([GitHub](https://github.com/ChernovAndrey), [LinkedIn](https://www.linkedin.com/in/andrei-chernov-58b157236/))

In this tutorial, you will build an LLM agent using the **MistralAI** language model and the **LlamaIndex** framework. The agent's purpose is to intelligently find and summarize research papers from **Arxiv** based on user queries. It will utilize three specialized tools to search a local knowledge base, fetch new papers, and download PDFs.

## Prerequisites & Setup

Before you begin, ensure you have the following:

1.  A **MistralAI API Key**. You can obtain one from the [MistralAI console](https://console.mistral.ai/api-keys/).
2.  (Optional) A **Phoenix API Key** for tracing and evaluation, available from [Arize Phoenix](https://app.phoenix.arize.com/login/sign-up).

First, install the required Python libraries.

```bash
pip install arxiv==2.1.3 llama_index==0.12.3 llama-index-llms-mistralai==0.3.0 llama-index-embeddings-mistralai==0.3.0
pip install arize-phoenix==7.2.0 arize-phoenix-evals==0.18.0 openinference-instrumentation-llama-index==3.0.2
```

Now, import the necessary modules and set up your API key.

```python
from getpass import getpass
import requests
import arxiv
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent

# Securely input your MistralAI API Key
api_key = getpass("Enter your MistralAI API Key: ")

# Initialize the LLM
llm = MistralAI(api_key=api_key, model='mistral-large-latest')

# Initialize the embedding model
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)
```

## Step 1: Define the Core Tools

The agent's intelligence comes from three tools it can use to accomplish tasks.

### Tool 1: Fetch Papers from Arxiv

This function searches Arxiv for recent papers on a given topic.

```python
def fetch_arxiv_papers(title: str, papers_count: int):
    """
    Fetches recent papers from Arxiv for a given topic.

    Args:
        title (str): The research topic to search for.
        papers_count (int): The maximum number of papers to fetch.

    Returns:
        list: A list of dictionaries containing paper metadata.
    """
    search_query = f'all:"{title}"'
    search = arxiv.Search(
        query=search_query,
        max_results=papers_count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    client = arxiv.Client()
    results = client.results(search)

    for result in results:
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
```

### Tool 2: Download a PDF

This function downloads a PDF from a given URL to a local file.

```python
def download_pdf(pdf_url: str, output_file: str):
    """
    Downloads a PDF file from a URL.

    Args:
        pdf_url (str): The URL of the PDF.
        output_file (str): The local path to save the file.

    Returns:
        str: A success or error message.
    """
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(output_file, "wb") as file:
            file.write(response.content)
        return f"PDF downloaded successfully and saved as '{output_file}'."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
```

## Step 2: Build a Knowledge Base with RAG

To give the agent immediate access to recent research, you will create a RAG (Retrieval-Augmented Generation) query engine from a set of papers.

First, fetch some initial papers to populate the knowledge base. To stay within free tier limits, we'll start with 10 papers on "Language Models".

```python
# Fetch initial papers
initial_papers = fetch_arxiv_papers("Language Models", 10)
```

Next, convert the paper metadata into `Document` objects that LlamaIndex can process.

```python
def create_documents_from_papers(papers):
    """Converts a list of paper dictionaries into LlamaIndex Document objects."""
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

documents = create_documents_from_papers(initial_papers)
```

Now, configure the text chunking settings and create a vector index from the documents.

```python
Settings.chunk_size = 1024
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
```

Indexing can be slow and costly. To avoid re-indexing, persist the index to disk and reload it.

```python
# Persist the index locally
index.storage_context.persist('index/')

# Reload the index from storage
storage_context = StorageContext.from_defaults(persist_dir='index/')
index = load_index_from_storage(storage_context, embed_model=embed_model)
```

Finally, create the RAG Query Engine tool from the index. Providing a clear name and description is crucial for the agent to understand when to use this tool.

```python
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="research_paper_query_engine_tool",
    description="A RAG engine containing recent research papers. Use this to search for papers within the stored knowledge base.",
)
```

## Step 3: Assemble the Tools and Create the Agent

Wrap the Python functions into tools the agent can use, and then instantiate the ReAct agent.

```python
# Create tools from the Python functions
download_pdf_tool = FunctionTool.from_defaults(
    download_pdf,
    name='download_pdf_file_tool',
    description='Downloads a PDF file from a provided URL to a local file.'
)

fetch_arxiv_tool = FunctionTool.from_defaults(
    fetch_arxiv_papers,
    name='fetch_from_arxiv',
    description='Fetches the {papers_count} most recent papers on the topic {title} from Arxiv.'
)

# Create the ReAct Agent with all three tools
agent = ReActAgent.from_tools([download_pdf_tool, rag_tool, fetch_arxiv_tool], llm=llm, verbose=True)
```

The ReAct agent works in a loop:
1.  **Reason:** It analyzes the query to decide if it can answer directly or needs to use a tool.
2.  **Act:** It executes the chosen tool.
3.  It repeats the **Reason** step with the new information from the tool's output until it can formulate a final answer.

## Step 4: Interact with the Agent

Let's test the agent with a query template designed to leverage its tools intelligently.

```python
# Define a query template
q_template = (
    "I am interested in {topic}. \n"
    "Find papers in your knowledge database related to this topic; use the following template to query research_paper_query_engine_tool tool: 'Provide title, summary, authors and link to download for papers related to {topic}'. If there are not, could you fetch the recent one from arXiv? \n"
)

# Ask about "Audio Models"
answer = agent.chat(q_template.format(topic="Audio Models"))
print(answer.response)
```

**Expected Agent Behavior:**
1.  The agent will first try to use the `research_paper_query_engine_tool` (RAG) to find papers on "Audio Models" in its local knowledge base.
2.  Since our initial knowledge base only contains papers on "Language Models", the RAG tool will likely find no matches.
3.  The agent will then reason that it should use the `fetch_from_arxiv` tool to get recent papers on "Audio Models".
4.  Finally, it will summarize the fetched papers for you.

Because the agent maintains chat history, you can make follow-up requests without repetition.

```python
# Request to download the papers it just mentioned
answer = agent.chat("Download the papers, which you mentioned above")
print(answer.response)
```

The agent will use the `download_pdf_file_tool` for each PDF URL it has, confirming the downloads.

## Step 5: Test a Query Within the Knowledge Base

Now, let's ask about a topic that *is* in our initial knowledge base.

```python
answer = agent.chat(q_template.format(topic="Multimodal Models"))
print(answer.response)
```

This time, the agent should successfully use the RAG tool to find and summarize relevant papers from the stored index without needing to fetch from Arxiv.

## (Optional) Step 6: Trace and Evaluate with Phoenix

To gain deep visibility into the agent's decision-making process, you can instrument it with Arize Phoenix for tracing and evaluation.

```python
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import os

# Securely input your Phoenix API Key
PHOENIX_API_KEY = getpass("Enter your Phoenix API Key: ")
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# Initialize tracing
tracer = register(project_name="arxiv-agentic-rag")
LlamaIndexInstrumentor().instrument()
```

After enabling instrumentation, run your agent queries again. You can then visit your [Phoenix dashboard](https://app.phoenix.arize.com) to inspect detailed traces of each step, including the agent's thoughts, tool calls, and final responses.

## Summary

You have successfully built a functional LLM agent that can:
*   Search a local knowledge base of research papers using RAG.
*   Dynamically fetch new papers from Arxiv when needed.
*   Download PDFs of papers upon request.
*   Reason through multi-step queries using the ReAct paradigm.

This agent provides a foundation for building more complex research assistants. You can extend it by adding more tools, connecting it to a proper vector database, or implementing more sophisticated evaluation metrics.