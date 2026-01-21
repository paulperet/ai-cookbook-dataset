# Building an LLM Agent to Find Relevant Research Papers from Arxiv

**Author:** Andrei Chernov ([GitHub](https://github.com/ChernovAndrey), [LinkedIn](https://www.linkedin.com/in/andrei-chernov-58b157236/))

This guide walks you through building an intelligent agent using the **MistralAI** language model and the **LlamaIndex** framework. The agent's purpose is to help you find, summarize, and download relevant research papers from **Arxiv** based on your queries. You will create an agent equipped with three specialized tools to handle different aspects of the research discovery process.

## Prerequisites & Setup

Before you begin, ensure you have the necessary libraries installed. You will also need a MistralAI API key.

### Step 1: Install Required Packages

Run the following command in your terminal or notebook environment to install the required Python packages.

```bash
pip install arxiv==2.1.3 llama_index==0.12.3 llama-index-llms-mistralai==0.3.0 llama-index-embeddings-mistralai==0.3.0
```

### Step 2: Import Libraries and Set Up the LLM

After installation, import the necessary modules and configure the MistralAI language model with your API key.

```python
from getpass import getpass
import requests
import arxiv
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent

# Securely input your MistralAI API key
api_key = getpass("Enter your MistralAI API Key: ")

# Initialize the LLM
llm = MistralAI(api_key=api_key, model='mistral-large-latest')
```

### Step 3: Configure the Embedding Model

To build a Retrieval-Augmented Generation (RAG) system, you need an embedding model to create vector representations of text. We'll use MistralAI's embedding model.

```python
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)
```

## Step 4: Fetch Initial Papers from Arxiv

First, you'll create a function to fetch recent papers from Arxiv. This function will serve as one of the agent's tools and also provide the initial dataset for the RAG knowledge base.

```python
def fetch_arxiv_papers(title: str, papers_count: int):
    """
    Fetches recent papers from Arxiv based on a search title.

    Args:
        title (str): The topic to search for.
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

# Fetch the last 10 papers on "Language Models" to stay within free API limits.
papers = fetch_arxiv_papers("Language Models", 10)
```

## Step 5: Prepare Documents for Indexing

LlamaIndex works with `Document` objects. You need to convert the fetched paper metadata into a structured text format suitable for indexing.

```python
def create_documents_from_papers(papers):
    """
    Converts a list of paper dictionaries into LlamaIndex Document objects.

    Args:
        papers (list): List of paper metadata dictionaries.

    Returns:
        list: A list of Document objects.
    """
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

# Create the document list
documents = create_documents_from_papers(papers)
```

## Step 6: Build and Persist the Vector Index

Now, create a vector index from the documents. This index enables efficient semantic search. To avoid re-indexing (which is costly), you will persist the index to disk.

```python
# Configure chunking settings for the index
Settings.chunk_size = 1024
Settings.chunk_overlap = 50

# Create the vector index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Persist the index to the 'index/' directory
index.storage_context.persist('index/')

# Reload the index from storage
storage_context = StorageContext.from_defaults(persist_dir='index/')
index = load_index_from_storage(storage_context, embed_model=embed_model)
```

## Step 7: Create the RAG Query Engine Tool

The first tool for your agent is a RAG query engine. It will search the indexed papers to answer questions. Providing a clear name and description helps the agent understand when to use this tool.

```python
# Create a query engine from the index
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

# Wrap the query engine as a tool
rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="research_paper_query_engine_tool",
    description="A RAG engine containing recent research papers. Use this to find papers within the stored knowledge base.",
)
```

## Step 8: Build the Remaining Tools

Your agent needs two more tools: one to download PDFs and another to fetch new papers directly from Arxiv if they aren't in the knowledge base.

### PDF Download Tool

```python
def download_pdf(pdf_url, output_file):
    """
    Downloads a PDF file from a given URL and saves it locally.

    Args:
        pdf_url (str): The URL of the PDF file.
        output_file (str): The local file path to save the PDF.

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

download_pdf_tool = FunctionTool.from_defaults(
    download_pdf,
    name='download_pdf_file_tool',
    description='Downloads a PDF file from a provided URL link.'
)
```

### Arxiv Fetch Tool

You will reuse the `fetch_arxiv_papers` function as a tool.

```python
fetch_arxiv_tool = FunctionTool.from_defaults(
    fetch_arxiv_papers,
    name='fetch_from_arxiv',
    description='Fetches the {papers_count} most recent papers on the topic {title} from Arxiv.'
)
```

## Step 9: Assemble the ReAct Agent

With all tools ready, you can now create the ReAct agent. This agent uses a Reason-Act loop: it reasons about the query, decides if a tool is needed, acts by using the tool, and then reasons again with the new information.

```python
# Create the agent with all three tools
agent = ReActAgent.from_tools([download_pdf_tool, rag_tool, fetch_arxiv_tool], llm=llm, verbose=True)
```

## Step 10: Interact with Your Agent

Let's test the agent with a predefined query template. This template instructs the agent to first search its knowledge base and, if necessary, fetch papers from Arxiv.

```python
# Define a query template
q_template = (
    "I am interested in {topic}. \n"
    "Find papers in your knowledge database related to this topic; use the following template to query research_paper_query_engine_tool tool: 'Provide title, summary, authors and link to download for papers related to {topic}'. If there are not, could you fetch the recent one from arXiv? \n"
)

# Ask about "Audio-Language Models"
answer = agent.chat(q_template.format(topic="Audio-Language Models"))
print(answer.response)
```

**Expected Output:**
The agent will use the RAG tool, find relevant papers from its indexed knowledge base, and return a summary including the title, authors, a brief summary, and a download link for the most relevant paper.

### Testing Agent Memory and Follow-up Actions

The agent maintains chat history. You can ask a follow-up question without repeating the context.

```python
# Request to download the paper mentioned in the previous answer
answer = agent.chat("Download the papers, which you mentioned above")
print(answer.response)
```

**Expected Output:**
The agent will use the `download_pdf_file_tool` to download the PDF and confirm the successful download.

### Testing with a Topic Outside the Knowledge Base

Let's see how the agent handles a query for a topic not present in the initial indexed papers.

```python
# Ask about "Gaussian process", a topic not in the initial fetch
answer = agent.chat(q_template.format(topic="Gaussian process"))
print(answer.response)
```

**Expected Output:**
The agent will first attempt to use the RAG tool. Finding no relevant papers, it will then use the `fetch_from_arxiv` tool to retrieve recent papers on Gaussian processes from Arxiv, summarize one, and provide a download link.

## Conclusion

You have successfully built an LLM-powered research assistant agent. This agent can:
1.  **Search** a local knowledge base of recent papers using RAG.
2.  **Fetch** new papers directly from Arxiv when needed.
3.  **Download** PDFs of papers for local storage.
4.  **Reason** about when to use each tool, thanks to the ReAct agent framework.

You can extend this agent by connecting the index to a proper vector database for scalability, adding more sophisticated tools, or customizing the prompts for different types of queries.