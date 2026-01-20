# Enhancing RAG Reasoning with Knowledge Graphs

_Authored by: [Diego Carpintero](https://github.com/dcarpintero)_

Knowledge Graphs provide a method for modeling and storing interlinked information in a format that is both human- and machine-understandable. These graphs consist of *nodes* and *edges*, representing entities and their relationships. Unlike traditional databases, the inherent expressiveness of graphs allows for richer semantic understanding, while providing the flexibility to accommodate new entity types and relationships without being constrained by a fixed schema.

By combining knowledge graphs with embeddings (vector search), we can leverage *multi-hop connectivity* and *contextual understanding of information* to enhance reasoning and explainability in LLMs. 

This notebook explores the practical implementation of this approach, demonstrating how to:
- Build a knowledge graph in [Neo4j](https://neo4j.com/docs/) related to research publications using a synthetic dataset,
- Project a subset of our data fields into a high-dimensional vector space using an [embedding model](https://python.langchain.com/v0.2/docs/integrations/text_embedding/),
- Construct a vector index on those embeddings to enable similarity search, and
- Extract insights from our graph using natural language by easily converting user queries into [cypher](https://neo4j.com/docs/cypher-manual/current/introduction/) statements with [LangChain](https://python.langchain.com/v0.2/docs/introduction/):

## Initialization

```python
%pip install neo4j langchain langchain_openai langchain-community python-dotenv --quiet
```

### Set up a Neo4j instance

We will create our Knowledge Graph using [Neo4j](https://neo4j.com/docs/), an open-source database management system that specializes in graph database technology.

For a quick and easy setup, you can start a free instance on [Neo4j Aura](https://neo4j.com/product/auradb/).

You might then set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` as environment variables using a `.env` file: 

```python
import dotenv
dotenv.load_dotenv('.env', override=True)
```

Langchain provides the `Neo4jGraph` class to interact with Neo4j:

```python
import os
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(
    url=os.environ['NEO4J_URI'], 
    username=os.environ['NEO4J_USERNAME'],
    password=os.environ['NEO4J_PASSWORD'],
)
```

### Loading Dataset into a Graph

The below example creates a connection with our `Neo4j` database and populates it with [synthetic data](https://github.com/dcarpintero/generative-ai-101/blob/main/dataset/synthetic_articles.csv) comprising research articles and their authors. 

The entities are: 
- *Researcher*
- *Article*
- *Topic*

Whereas the relationships are:
- *Researcher* --[PUBLISHED]--> *Article*
- *Article* --[IN_TOPIC]--> *Topic*

```python
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

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

graph.query(q_load_articles)
```

Let's check that the nodes and relationships have been initialized correctly:

```python
graph.refresh_schema()
print(graph.get_schema)
```

Our knowledge graph can be inspected in the Neo4j workspace:

### Building a Vector Index

Now we construct a vector index to efficiently search for relevant *articles* based on their *topic, title, and abstract*. This process involves calculating the embeddings for each article using these fields. At query time, the system finds the most similar articles to the user's input by employing a similarity metric, such as cosine distance.

```python
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=os.environ['NEO4J_URI'],
    username=os.environ['NEO4J_USERNAME'],
    password=os.environ['NEO4J_PASSWORD'],
    index_name='articles',
    node_label="Article",
    text_node_properties=['topic', 'title', 'abstract'],
    embedding_node_property='embedding',
)
```

**Note:** To access OpenAI embedding models you will need to create an OpenAI account, get an API key, and set `OPENAI_API_KEY` as an environment variable. You might also find it useful to experiment with another [embedding model](https://python.langchain.com/v0.2/docs/integrations/text_embedding/) integration.

## Q&A on Similarity

`Langchain RetrievalQA` creates a question-answering (QA) chain using the above vector index as a retriever.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vector_index.as_retriever()
)
```

Let's ask '*which articles discuss how AI might affect our daily life?*':

```python
r = vector_qa.invoke(
    {"query": "which articles discuss how AI might affect our daily life? include the article titles and abstracts."}
)
print(r['result'])
```

## Traversing Knowledge Graphs for Inference

Knowledge graphs are excellent for making connections between entities, enabling the extraction of patterns and the discovery of new insights.

This section demonstrates how to implement this process and integrate the results into an LLM pipeline using natural language queries.

### Graph-Cypher-Chain w/ LangChain

To construct expressive and efficient queries `Neo4j` users `Cypher`, a declarative query language inspired by SQL. `LangChain` provides the wrapper `GraphCypherQAChain`, an abstraction layer that allows querying graph databases using natural language, making it easier to integrate graph-based data retrieval into LLM pipelines.

In practice, `GraphCypherQAChain`:
- generates Cypher statements (queries for graph databases like Neo4j) from user input (natural language) applying in-context learning (prompt engineering),
- executes said statements against a graph database, and 
- provides the results as context to ground the LLM responses on accurate, up-to-date information:

**Note:** This implementation involves executing model-generated graph queries, which carries inherent risks such as unintended access or modification of sensitive data in the database. To mitigate these risks, ensure that your database connection permissions are as restricted as possible to meet the specific needs of your chain/agent. While this approach reduces risk, it does not eliminate it entirely.

```python
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

graph.refresh_schema()

cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm = ChatOpenAI(temperature=0, model_name='gpt-4o'),
    qa_llm = ChatOpenAI(temperature=0, model_name='gpt-4o'), 
    graph=graph,
    verbose=True,
)
```

### Query Samples using Natural Language

Note in the following examples how the results from the cypher query execution are provided as context to the LLM:

#### **"*How many articles has published Emily Chen?*"**

In this example, our question '*How many articles has published Emily Chen?*' will be translated into the Cyper query:

```
MATCH (r:Researcher {name: "Emily Chen"})-[:PUBLISHED]->(a:Article)
RETURN COUNT(a) AS numberOfArticles
```

which matches nodes labeled `Author` with the name 'Emily Chen' and traverses the `PUBLISHED` relationships to `Article` nodes. 
It then counts the number of `Article` nodes connected to 'Emily Chen':

```python
# the answer should be '7'
cypher_chain.invoke(
    {"query": "How many articles has published Emily Chen?"}
)
```

[Generated Cypher: ..., Full Context: [{'numberOfArticles': 7}], ..., Finished chain.]

#### **"*Are there any pair of researchers who have published more than three articles together?*"**

In this example, the query '*are there any pair of researchers who have published more than three articles together?*' results in the Cypher query:

```
MATCH (r1:Researcher)-[:PUBLISHED]->(a:Article)<-[:PUBLISHED]-(r2:Researcher)
WHERE r1 <> r2
WITH r1, r2, COUNT(a) AS sharedArticles
WHERE sharedArticles > 3
RETURN r1.name, r2.name, sharedArticles
```

which results in traversing from the `Researcher` nodes to the `PUBLISHED` relationship to find connected `Article` nodes, and then traversing back to find `Researchers` pairs.

```python
# the answer should be David Johnson & Emily Chen, Robert Taylor & Emily Chen
cypher_chain.invoke(
    {"query": "are there any pair of researchers who have published more than three articles together?"}
)
```

[Generated Cypher: ..., Full Context: [{'r1.name': 'David Johnson', 'r2.name': 'Emily Chen', 'sharedArticles': 4}, ..., {'r1.name': 'Emily Chen', 'r2.name': 'Robert Taylor', 'sharedArticles': 4}], ..., Finished chain.]

#### **"*which researcher has collaborated with the most peers?*"**

Let's find out who is the researcher with most peers collaborations. 
Our query '*which researcher has collaborated with the most peers?*' results now in the Cyper:

```
MATCH (r:Researcher)-[:PUBLISHED]->(:Article)<-[:PUBLISHED]-(peer:Researcher)
WITH r, COUNT(DISTINCT peer) AS peerCount
RETURN r.name AS researcher, peerCount
ORDER BY peerCount DESC
LIMIT 1
```

Here, we need to start from all `Researcher` nodes and traverse their `PUBLISHED` relationships to find connected `Article` nodes. For each `Article` node, Neo4j then traverses back to find other `Researcher` nodes (peer) who have also published the same article.

```python
# the answer should be 'David Johnson'
cypher_chain.invoke(
    {"query": "Which researcher has collaborated with the most peers?"}
)
```

[Generated Cypher: ..., Full Context: [{'researcher': 'David Johnson', 'collaborators': 6}], ..., Finished chain.]

----