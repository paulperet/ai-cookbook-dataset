# Adaptive RAG with LangGraph and Mistral

This guide walks you through building an Adaptive Retrieval-Augmented Generation (RAG) system using LangGraph and Mistral. Adaptive RAG intelligently routes user queries between a local knowledge base and web search based on complexity and relevance, while implementing quality checks to ensure grounded and useful responses.

## Prerequisites

Ensure you have the required API keys and install the necessary packages.

### 1. Install Dependencies

```bash
pip install --quiet -U langchain langchain_community tiktoken langchain-mistralai scikit-learn langgraph tavily-python bs4
```

### 2. Set Environment Variables

Set your Mistral and Tavily API keys. Optionally, configure LangSmith for tracing.

```python
import os
import getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Required API keys
_set_env("MISTRAL_API_KEY")
_set_env("TAVILY_API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Optional: LangSmith for tracing
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mistral-cookbook"
```

## Step 1: Build a Knowledge Base

First, we'll create a vector store from three blog posts about AI topics.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_mistralai import MistralAIEmbeddings

# URLs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load and split documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

# Create vector store and retriever
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=MistralAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
```

## Step 2: Configure the Core LLM

We'll use Mistral's function calling capability to produce structured outputs for routing and grading tasks.

```python
from langchain_mistralai import ChatMistralAI

mistral_model = "mistral-large-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0)
```

## Step 3: Create Specialized Grading and Routing Components

These components use structured outputs to make decisions within the RAG workflow.

### 3.1 Query Router

This component decides whether a query should be answered from the vector store or via web search.

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, use web-search."""
```

### 3.2 Document Relevance Grader

This grades whether a retrieved document is relevant to the user's question.

```python
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)

doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

doc_grader_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"
```

### 3.3 Generation Component

This is the standard RAG generation step.

```python
from langchain_core.output_parsers import StrOutputParser

rag_prompt = """You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.

If you don't know the answer, just say that you don't know.

Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

### 3.4 Hallucination Grader

This checks if the generated answer is grounded in the provided facts.

```python
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
    explanation: str = Field(description="Explain the reasoning for the score")

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

hallucination_grader_instructions = """You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = "FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}"
```

### 3.5 Answer Grader

This evaluates whether the generated answer is useful and addresses the original question.

```python
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
    explanation: str = Field(description="Explain the reasoning for the score")

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

answer_grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION

(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Score:

A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = "QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}"
```

## Step 4: Configure Web Search Tool

For queries routed to web search, we'll use the Tavily search engine.

```python
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)
```

## Step 5: Build the LangGraph Workflow

We'll now construct the adaptive RAG system as a stateful graph.

### 5.1 Define the Graph State

The state is a dictionary that propagates information between nodes.

```python
import operator
from typing_extensions import TypedDict
from typing import List, Annotated

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    web_search : str # Binary decision to run web search
    max_retries : int # Max number of retries for answer generation
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents : List[str] # List of retrieved documents
```

### 5.2 Define Graph Nodes

Each node is a function that receives and modifies the state.

#### Retrieve Node

```python
from langchain.schema import Document

def retrieve(state):
    """
    Retrieve documents from vectorstore
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}
```

#### Generate Node

```python
from langchain_core.messages import HumanMessage

def generate(state):
    """
    Generate answer using RAG on retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step+1}
```

#### Grade Documents Node

```python
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        score = structured_llm_doc_grader.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}
```

*(Note: The original code snippet for `grade_documents` was cut off. The full implementation would include returning the state with updated `documents` and `web_search` keys.)*

## Next Steps

To complete the Adaptive RAG system, you would continue by:

1.  Defining additional nodes for web search, hallucination grading, and answer grading.
2.  Creating the graph edges that control the flow between nodes based on conditions (e.g., `web_search == "Yes"`).
3.  Compiling the graph and running it with user questions.

This architecture provides a robust, self-correcting RAG pipeline that dynamically chooses data sources and validates its own outputs for relevance, grounding, and usefulness.