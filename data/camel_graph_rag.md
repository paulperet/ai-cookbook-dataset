# Building a Knowledge Graph with CAMEL AI and Mistral Models

This guide demonstrates how to use the CAMEL AI framework with Mistral's advanced language models to perform Graph-based Retrieval-Augmented Generation (RAG). You will learn to extract structured knowledge from text, store it in a Neo4j graph database, and query it using a hybrid approach that combines vector and graph retrieval.

## Prerequisites and Setup

### 1. Install the CAMEL Package
Begin by installing the required package with all dependencies.
```bash
pip install camel-ai[all]==0.1.6.0
```

### 2. Import Required Modules
Import the necessary components from the CAMEL library.
```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, OllamaConfig
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.retrievers import AutoRetriever
from camel.embeddings import MistralEmbedding
from camel.types import StorageType, RoleType
from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.messages import BaseMessage
```

### 3. Configure API Keys
Securely set your Mistral AI API key as an environment variable.
```python
import os
from getpass import getpass

mistral_api_key = getpass('Enter your Mistral API key: ')
os.environ["MISTRAL_API_KEY"] = mistral_api_key
```

### 4. Connect to Neo4j
Set up a connection to your Neo4j database instance. Ensure you have your URI, username, and password ready.
```python
n4j = Neo4jGraph(
    url="Your_URI",
    username="Your_Username",
    password="Your_Password",
)
```

## Step 1: Initialize the Language Model

Create an instance of the Mistral Large 2 model using CAMEL's `ModelFactory`. This model will power the knowledge extraction and generation tasks.

```python
# Initialize the Mistral Large 2 model via the API
mistral_large_2 = ModelFactory.create(
    model_platform=ModelPlatformType.MISTRAL,
    model_type=ModelType.MISTRAL_LARGE,
    model_config_dict=MistralConfig(temperature=0.2).__dict__,
)
```

> **Note:** You can also run a local model using Ollama. Uncomment and use the following configuration if preferred:
> ```python
> mistral_large_2_local = ModelFactory.create(
>     model_platform=ModelPlatformType.OLLAMA,
>     model_type="mistral-large",
>     model_config_dict=OllamaConfig(temperature=0.2).__dict__,
> )
> ```

## Step 2: Extract a Knowledge Graph from Text

### 2.1 Initialize the Knowledge Graph Agent
Set up the `KnowledgeGraphAgent` and a text loader (`UnstructuredIO`).

```python
uio = UnstructuredIO()
kg_agent = KnowledgeGraphAgent(model=mistral_large_2)
```

### 2.2 Provide Example Text
Define a sample text for the agent to process.

```python
text_example = """
CAMEL has developed a knowledge graph agent can run with Mistral AI's most
advanced model, the Mistral Large 2. This knowledge graph agent is capable
of extracting entities and relationships from given content and create knowledge
graphs automatically.
"""
```

### 2.3 Create and Process a Text Element
Convert the text into a structured element and pass it to the agent.

```python
# Create an element from the text
element_example = uio.create_element_from_text(text=text_example)

# Let the agent extract node and relationship information
ans_element = kg_agent.run(element_example, parse_graph_elements=False)
print(ans_element)
```

The agent will analyze the text and output a structured breakdown. For the provided example, it identifies entities like "CAMEL" (Organization), "knowledge graph agent" (Software), and "Mistral Large 2" (Model), along with their relationships (e.g., "Developed", "RunsWith").

### 2.4 Parse into Graph Elements
To get the data in a format ready for the database, run the agent with `parse_graph_elements=True`.

```python
graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
print(graph_elements)
```

This returns a `GraphElement` object containing lists of `Node` and `Relationship` objects, each with properties.

### 2.5 Store the Graph in Neo4j
Add the extracted graph structure to your Neo4j database.

```python
n4j.add_graph_elements(graph_elements=[graph_elements])
```

You can now visualize this graph in your Neo4j browser.

## Step 3: Perform Hybrid Graph RAG

This section shows how to answer a complex query by combining vector search (for semantic similarity) with knowledge graph lookup (for precise relationships).

### 3.1 Set Up a Vector Retriever
Configure a retriever that uses local storage and Mistral's embedding model.

```python
camel_retriever = AutoRetriever(
    vector_storage_local_path="local_data/embedding_storage",
    storage_type=StorageType.QDRANT,
    embedding_model=MistralEmbedding(),
)
```

### 3.2 Define a User Query
Formulate the question you want the system to answer.

```python
query = "what's the relationship between Mistral Large 2 and Mistral AI? What kind of feature does Mistral Large 2 have?"
```

### 3.3 Perform Vector Retrieval
Use the retriever to find semantically relevant text from a source. Here, we use a Mistral AI news article as the source content.

```python
vector_result = camel_retriever.run_vector_retriever(
    query=query,
    content_input_paths="https://mistral.ai/news/mistral-large-2407/",
)
print(vector_result)
```

The output will be the original query followed by the most relevant text chunks retrieved.

### 3.4 Enrich the Knowledge Graph with Source Content
Parse the full source article, chunk it, and use the Knowledge Graph Agent to extract all entities and relationships, adding them to Neo4j.

```python
# Parse content from the Mistral website
elements = uio.parse_file_or_url(
    input_path="https://mistral.ai/news/mistral-large-2407/"
)
# Chunk the content by title for processing
chunk_elements = uio.chunk_elements(
    chunk_type="chunk_by_title", elements=elements
)

# Extract and store graph elements from each chunk
graph_elements = []
for chunk in chunk_elements:
    graph_element = kg_agent.run(chunk, parse_graph_elements=True)
    n4j.add_graph_elements(graph_elements=[graph_element])
    graph_elements.append(graph_element)
```

### 3.5 Query the Knowledge Graph
Extract entities from the user's query and find their connections within the graph database.

```python
# Create an element from the user query
query_element = uio.create_element_from_text(text=query)
# Extract graph elements from the query
ans_element = kg_agent.run(query_element, parse_graph_elements=True)

# For each node found in the query, search for its relationships in Neo4j
kg_result = []
for node in ans_element.nodes:
    n4j_query = f"""
MATCH (n {{id: '{node.id}'}})-[r]->(m)
RETURN 'Node ' + n.id + ' (label: ' + labels(n)[0] + ') has relationship ' + type(r) + ' with Node ' + m.id + ' (label: ' + labels(m)[0] + ')' AS Description
UNION
MATCH (n)<-[r]-(m {{id: '{node.id}'}})
RETURN 'Node ' + m.id + ' (label: ' + labels(m)[0] + ') has relationship ' + type(r) + ' with Node ' + n.id + ' (label: ' + labels(n)[0] + ')' AS Description
"""
    result = n4j.query(query=n4j_query)
    kg_result.extend(result)

kg_result = [item['Description'] for item in kg_result]
print(kg_result)
```

This will return a list of relationship descriptions from the graph, such as `'Node Mistral Large 2 (label: Model) has relationship HASFEATURE with Node 128k context window (label: Feature)'`.

### 3.6 Combine Retrieval Results
Merge the context from the vector search and the knowledge graph search to provide a comprehensive information base.

```python
combined_results = vector_result + "\n".join(kg_result)
```

### 3.7 Generate the Final Answer
Use a `ChatAgent` to synthesize an answer based on the combined retrieved context.

```python
# Configure the assistant agent
sys_msg = BaseMessage.make_assistant_message(
    role_name="CAMEL Agent",
    content="""You are a helpful assistant to answer questions.
        I will give you the Original Query and Retrieved Context.
        Answer the Original Query based on the Retrieved Context.""",
)

camel_agent = ChatAgent(system_message=sys_msg, model=mistral_large_2)

# Formulate the prompt with the query and combined context
user_prompt = f"""
The Original Query is {query}
The Retrieved Context is {combined_results}
"""

user_msg = BaseMessage.make_user_message(
    role_name="CAMEL User", content=user_prompt
)

# Get the agent's response
agent_response = camel_agent.step(user_msg)
print(agent_response.msg.content)
```

The final output will be a coherent answer detailing the relationship between Mistral AI and Mistral Large 2, along with a list of the model's key features, all grounded in the retrieved evidence.

## Summary

This tutorial demonstrated a complete workflow for Graph RAG using CAMEL AI and Mistral models:

1.  **Setup:** Installed dependencies, configured API keys, and connected to Neo4j.
2.  **Knowledge Extraction:** Used the `KnowledgeGraphAgent` to automatically identify entities and relationships from text and store them in a graph database.
3.  **Hybrid Retrieval:** Answered a complex query by combining:
    *   **Vector Search:** For finding semantically relevant text passages.
    *   **Graph Search:** For retrieving precise, structured relationships between entities.
4.  **Answer Synthesis:** Leveraged a `ChatAgent` to generate a final, context-rich answer from the combined retrieval results.

This approach leverages the strengths of both unstructured text retrieval and structured knowledge graphs, enabling powerful and accurate question-answering systems.