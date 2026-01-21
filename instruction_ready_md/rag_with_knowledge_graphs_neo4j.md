# Enhancing RAG Reasoning with Knowledge Graphs: A Step-by-Step Guide

_Authored by: [Diego Carpintero](https://github.com/dcarpintero)_

Knowledge Graphs (KGs) provide a powerful method for modeling and storing interlinked information in a format that is both human- and machine-understandable. By combining the semantic expressiveness of graphs with the similarity search capabilities of vector embeddings, we can build systems that leverage *multi-hop connectivity* and *contextual understanding* to significantly enhance reasoning and explainability in Large Language Models (LLMs).

This guide walks you through the practical implementation of this approach. You will learn how to:
1.  Build a knowledge graph in Neo4j using a synthetic dataset of research publications.
2.  Project article data into a vector space and create a search index.
3.  Query the graph using natural language by converting questions into Cypher statements with LangChain.

## Prerequisites & Setup

Before you begin, ensure you have the necessary Python libraries installed.

```bash
pip install neo4j langchain langchain_openai langchain_community python-dotenv --quiet
```

You will also need access to:
1.  A **Neo4j Database Instance**: You can quickly start a free instance on [Neo4j Aura](https://neo4j.com/product/auradb/).
2.  An **OpenAI API Key**: Required for generating embeddings and powering the LLM. Sign up at [OpenAI](https://platform.openai.com/).

Once you have your Neo4j credentials and OpenAI API key, create a `.env` file in your project directory to store them securely:

```plaintext
NEO4J_URI=your_neo4j_uri_here
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
OPENAI_API_KEY=your_openai_api_key_here
```

## Step 1: Connect to Your Neo4j Graph

First, we'll load our environment variables and establish a connection to the Neo4j database using LangChain's `Neo4jGraph` class.

```python
import os
from langchain_community.graphs import Neo4jGraph
import dotenv

# Load environment variables from the .env file
dotenv.load_dotenv('.env', override=True)

# Create a connection to the Neo4j graph database
graph = Neo4jGraph(
    url=os.environ['NEO4J_URI'],
    username=os.environ['NEO4J_USERNAME'],
    password=os.environ['NEO4J_PASSWORD'],
)
```

## Step 2: Populate the Graph with Data

We will populate our graph with a synthetic dataset of research articles. Our graph schema will include:
*   **Nodes (Entities):** `Researcher`, `Article`, `Topic`
*   **Relationships:** `PUBLISHED` (Researcher -> Article), `IN_TOPIC` (Article -> Topic)

The following Cypher query loads the data from a CSV file and creates the corresponding nodes and relationships in the database.

```python
# Define the Cypher query to load and structure the data
q_load_articles = """
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/dcarpintero/generative-ai-101/main/dataset/synthetic_articles.csv'
AS row
FIELDTERMINATOR ';'
MERGE (a:Article {title:row.Title})
SET a.abstract = row.Abstract,
    a.publication_date = date(row.Publication_Date)
FOREACH (researcher in split(row.Authors, ',') |
    MERGE (p:Researcher {name:trim(researcher)})
    MERGE (p)-[:PUBLISHED]->(a))
FOREACH (topic in [row.Topic] |
    MERGE (t:Topic {name:trim(topic)})
    MERGE (a)-[:IN_TOPIC]->(t))
"""

# Execute the query to populate the graph
graph.query(q_load_articles)
```

Let's verify the graph schema has been created correctly.

```python
# Refresh and print the graph schema
graph.refresh_schema()
print(graph.get_schema)
```

## Step 3: Build a Vector Search Index

To enable semantic search over the articles, we need to create vector embeddings for their content (topic, title, and abstract) and build an index. We'll use LangChain's `Neo4jVector` integration, which handles both the embedding generation and index creation.

```python
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

# Create a vector index from the existing graph data
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(), # Uses the OpenAIEmbeddings model (requires OPENAI_API_KEY)
    url=os.environ['NEO4J_URI'],
    username=os.environ['NEO4J_USERNAME'],
    password=os.environ['NEO4J_PASSWORD'],
    index_name='articles', # Name of the vector index in Neo4j
    node_label="Article", # Which nodes to index
    text_node_properties=['topic', 'title', 'abstract'], # Properties to embed
    embedding_node_property='embedding', # Property to store the embedding vector
)
```

**Note:** This step requires a valid `OPENAI_API_KEY`. You can also experiment with other [embedding model integrations](https://python.langchain.com/v0.2/docs/integrations/text_embedding/) provided by LangChain.

## Step 4: Perform Q&A Using Vector Similarity

With our vector index ready, we can create a simple question-answering chain that retrieves relevant articles based on semantic similarity to a user's query.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create a RetrievalQA chain using the vector index as the retriever
vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), # LLM to generate the final answer
    chain_type="stuff", # Simple method for stuffing context into the prompt
    retriever=vector_index.as_retriever()
)
```

Now, let's test it. We'll ask a question about articles discussing AI's impact on daily life.

```python
# Query the vector-based QA system
response = vector_qa.invoke(
    {"query": "which articles discuss how AI might affect our daily life? include the article titles and abstracts."}
)
print(response['result'])
```

The chain will retrieve the most semantically similar articles from the vector index and use the LLM to formulate a coherent answer based on their titles and abstracts.

## Step 5: Query the Knowledge Graph with Natural Language

While vector search is great for semantic similarity, knowledge graphs excel at traversing connections and extracting precise, structured information. LangChain's `GraphCypherQAChain` allows us to query the graph using natural language by automatically generating and executing Cypher queries.

**Important Security Note:** This chain executes model-generated database queries. Always restrict your database connection permissions to the minimum required for your application to mitigate risks of unintended data access or modification.

```python
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Ensure the graph schema is up-to-date
graph.refresh_schema()

# Create the GraphCypherQAChain
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-4o'), # LLM to generate Cypher
    qa_llm=ChatOpenAI(temperature=0, model_name='gpt-4o'), # LLM to generate final answer
    graph=graph,
    verbose=True, # Set to True to see the generated Cypher and context
)
```

### Example Queries and Graph Traversal

Let's see the chain in action with a few example questions. The chain will 1) generate a Cypher query, 2) run it against the database, and 3) use the results as context for the final LLM answer.

#### **Example 1: Counting Publications**

**Query:** "How many articles has published Emily Chen?"

```python
result = cypher_chain.invoke({"query": "How many articles has published Emily Chen?"})
print(result['result'])
```

*   **Generated Cypher Logic:** Finds the `Researcher` node for "Emily Chen" and counts all connected `Article` nodes via `PUBLISHED` relationships.
*   **Expected Answer:** `7`

#### **Example 2: Finding Collaborative Pairs**

**Query:** "Are there any pair of researchers who have published more than three articles together?"

```python
result = cypher_chain.invoke({"query": "are there any pair of researchers who have published more than three articles together?"})
print(result['result'])
```

*   **Generated Cypher Logic:** Finds pairs of distinct `Researcher` nodes connected to the same `Article` nodes, counts their shared articles, and filters for counts greater than 3.
*   **Expected Insight:** The answer should identify pairs like David Johnson & Emily Chen, and Robert Taylor & Emily Chen.

#### **Example 3: Identifying the Most Collaborative Researcher**

**Query:** "Which researcher has collaborated with the most peers?"

```python
result = cypher_chain.invoke({"query": "Which researcher has collaborated with the most peers?"})
print(result['result'])
```

*   **Generated Cypher Logic:** For each researcher, finds all unique co-authors (peers) via shared articles, counts them, and returns the researcher with the highest count.
*   **Expected Answer:** `David Johnson`

## Conclusion

You have successfully built a hybrid retrieval system that combines the strengths of vector similarity search and knowledge graph traversal. This approach provides a robust foundation for advanced RAG applications, offering both semantic understanding and the ability to reason over explicit relationships within your data.

To extend this project, consider:
*   Implementing a **hybrid retriever** that merges results from both the vector index and the graph chain.
*   Adding **multi-hop reasoning** where the LLM iteratively queries the graph to explore deeper connections.
*   Using the graph structure to **improve response explainability** by showing the paths of relationships used to reach an answer.