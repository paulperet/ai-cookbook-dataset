# Building and Querying Property Graphs with LlamaIndex

This guide walks you through constructing and querying a Property Graph Index in LlamaIndex. A property graph is a structured knowledge base consisting of labeled nodes (entities) with properties, connected by relationships to form paths (triplets). You'll learn how to extract this structured information from your documents and build powerful, multi-strategy retrievers to query it.

## Prerequisites

First, ensure you have the necessary libraries installed and imported.

```bash
pip install llama-index
```

```python
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
    SchemaLLMPathExtractor,
    LLMSynonymRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
    CypherTemplateRetriever,
    PGRetriever
)
from typing import List, Tuple, Literal
from pydantic.v1 import BaseModel, Field
```

## Step 1: Constructing the Property Graph Index

The core of this system is the `PropertyGraphIndex`. You build it by applying one or more **knowledge graph extractors** to your documents. These extractors parse your text to identify entities and the relationships between them.

### 1.1 Basic Index Construction

To create an index, provide your documents. By default, it uses `SimpleLLMPathExtractor` and `ImplicitPathExtractor`.

```python
# Assuming `documents` is a list of your loaded document objects
index = PropertyGraphIndex.from_documents(
    documents,
)
```

### 1.2 Using Custom Extractors

You can explicitly define which extractors to use for more control. All provided extractors will be applied to each document chunk.

```python
# Create your extractors first (examples in the next steps)
extractor1 = SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=10)
extractor2 = ImplicitPathExtractor()

# Build the index with your custom extractors
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[extractor1, extractor2, ...],
)
```

You can also insert documents or nodes after the index is created.

```python
index.insert(new_document)
# or
index.insert_nodes(list_of_nodes)
```

## Step 2: Configuring Knowledge Graph Extractors

Extractors are responsible for parsing structured paths (subject, predicate, object) from your text. Let's explore the available options.

### 2.1 SimpleLLMPathExtractor

This extractor uses an LLM to identify simple, single-hop relationships from the text.

```python
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,           # Your LLM instance
    max_paths_per_chunk=10,
    num_workers=4,     # For parallel processing
    show_progress=False,
)
```

#### Customizing the Extraction

You can fully customize the prompt and the parsing logic.

```python
prompt = (
    "Some text is provided below. Given the text, extract up to "
    "{max_paths_per_chunk} "
    "knowledge triples in the form of `subject,predicate,object` on each line. Avoid stopwords.\n"
)

def parse_fn(response_str: str) -> List[Tuple[str, str, str]]:
    """Parse the LLM's text response into a list of triples."""
    lines = response_str.strip().split("\n")
    triples = [line.split(",") for line in lines if line]
    return triples

custom_extractor = SimpleLLMPathExtractor(
    llm=llm,
    extract_prompt=prompt,
    parse_fn=parse_fn,
)
```

### 2.2 ImplicitPathExtractor

This extractor doesn't use an LLM. Instead, it parses relationships that are already defined in the `node.relationships` attribute of your LlamaIndex node objects. It's useful when your documents are pre-annotated.

```python
kg_extractor = ImplicitPathExtractor()
```

### 2.3 SchemaLLMPathExtractor

For strict, validated knowledge graphs, use this extractor. It forces the LLM to adhere to a predefined schema of allowed entities and relationships.

```python
# 1. Define allowed entities and relations using Literal types
entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["PART_OF", "HAS", "WORKED_AT"]

# 2. Define the valid connections between them
schema = [
    ("PLACE", "HAS", "PERSON"),
    ("PERSON", "PART_OF", "PLACE"),
    ("PERSON", "WORKED_AT", "ORGANIZATION")
]

# 3. Create the extractor
kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=schema,
    strict=True,  # If False, triples outside the schema are allowed
    max_paths_per_chunk=10,
)
```

## Step 3: Querying with Retrievers

Once your index is built, you can query it using retrievers. The power of the `PropertyGraphIndex` lies in its ability to **combine multiple retrieval strategies** for comprehensive results.

### 3.1 Creating a Retriever or Query Engine

You can create a standalone retriever or a full query engine (which uses a retriever and an LLM synthesizer).

```python
# Create a retriever that combines multiple strategies
retriever = index.as_retriever(
    include_text=True,   # Include the original source text with results
    similarity_top_k=2,
    sub_retrievers=[retriever1, retriever2, ...] # Combined strategies
)

# Or create a full query engine
query_engine = index.as_query_engine(
    include_text=True,
    similarity_top_k=2,
    sub_retrievers=[retriever1, retriever2, ...]
)

# Use them
retrieved_nodes = retriever.retrieve("Your query here")
response = query_engine.query("Your query here")
```

If you don't specify `sub_retrievers`, the default retrievers (`LLMSynonymRetriever` and `VectorContextRetriever` if embeddings are enabled) will be used.

### 3.2 LLMSynonymRetriever

This retriever expands your query by generating synonyms and related keywords using an LLM, then finds graph nodes matching those terms.

```python
synonym_retriever = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    include_text=False,      # Don't attach source text to results
    max_keywords=10,         # Max number of synonyms to generate
    path_depth=1,            # How many relationship hops to follow from matched nodes
)

# Use it as a sub-retriever
retriever = index.as_retriever(sub_retrievers=[synonym_retriever])
```

#### Customizing Synonym Generation

You can customize the prompt and output parsing.

```python
prompt = (
    "Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
    "Note, result should be in one-line, separated by '^' symbols."
    "----\n"
    "QUERY: {query_str}\n"
    "----\n"
    "KEYWORDS: "
)

def parse_fn(output: str) -> list[str]:
    matches = output.strip().split("^")
    return [x.strip().capitalize() for x in matches if x.strip()]

custom_synonym_retriever = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    synonym_prompt=prompt,
    output_parsing_fn=parse_fn,
)
```

### 3.3 VectorContextRetriever

This retriever uses vector similarity search to find relevant graph nodes. It requires an embedding model.

```python
vector_retriever = VectorContextRetriever(
    index.property_graph_store,
    embed_model=embed_model,     # Your embedding model
    include_text=False,
    similarity_top_k=2,          # Number of similar nodes to fetch
    path_depth=1,
)

retriever = index.as_retriever(sub_retrievers=[vector_retriever])
```

> **Note on Vector Stores:** If your graph store (e.g., Neo4j) supports vector queries natively, you only need the graph store. Otherwise, you must also provide a `vector_store` parameter (by default, it uses an in-memory `SimpleVectorStore`).

### 3.4 TextToCypherRetriever

For graph databases that support Cypher (like Neo4j), this retriever uses an LLM to translate a natural language query directly into a Cypher query, executes it, and returns the results.

> **Important:** The default `SimplePropertyGraphStore` does **not** support Cypher queries.

```python
# First, inspect your graph schema to inform prompt engineering
schema_str = index.property_graph_store.get_schema_str()
print(schema_str)

cypher_retriever = TextToCypherRetriever(
    index.property_graph_store,
    llm=llm,
    # Optional: Customize the template that instructs the LLM.
    # Must accept `schema` and `question` variables.
    # text_to_cypher_template=custom_template,
)
```

### 3.5 CypherTemplateRetriever

A more controlled alternative to `TextToCypherRetriever`. You provide a parameterized Cypher query template, and the LLM's role is only to fill in the parameter values.

```python
# 1. Define your Cypher query with a parameter (e.g., $names)
cypher_query = """
MATCH (c:Chunk)-[:MENTIONS]->(o)
WHERE o.name IN $names
RETURN c.text, o.name, o.label;
"""

# 2. Define a Pydantic model for the parameters.
# The field name (`names`) must match the Cypher parameter.
class TemplateParams(BaseModel):
    """Template params for a cypher query."""
    names: list[str] = Field(
        description="A list of entity names or keywords to use for lookup in a knowledge graph."
    )

# 3. Create the retriever
template_retriever = CypherTemplateRetriever(
    index.property_graph_store,
    TemplateParams,
    cypher_query
)

# 4. Use it. The LLM will infer the `names` list from the user's query.
retriever = index.as_retriever(sub_retrievers=[template_retriever])
```

## Step 4: Combining Multiple Retrievers

The true strength of the property graph index is combining these strategies. The `PGRetriever` (or `index.as_retriever`) merges results from all sub-retrievers.

```python
# Build individual retrievers
synonym_retriever = LLMSynonymRetriever(index.property_graph_store, llm=llm)
vector_retriever = VectorContextRetriever(index.property_graph_store, embed_model=embed_model)

# Combine them into a single retriever
combined_retriever = PGRetriever(
    sub_retrievers=[synonym_retriever, vector_retriever]
)

# Or use the index's convenience method (which does the same)
retriever = index.as_retriever(sub_retrievers=[synonym_retriever, vector_retriever])

# Now query
results = retriever.retrieve("Who worked at OpenAI?")
```

## Step 5: Persisting and Loading the Index

To save your constructed graph and its associated vector stores for later use:

```python
# Save to disk
index.storage_context.persist(persist_dir="./storage")

# Load it back later
storage_context = StorageContext.from_defaults(persist_dir="./storage")
loaded_index = load_index_from_storage(storage_context)
```

You can also build an index directly from existing graph and vector stores.

```python
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    vector_store=vector_store,
    llm=llm
)
```

## Summary

You've now learned the complete workflow for property graphs in LlamaIndex:
1.  **Extract** structured paths from documents using LLM-based or schema-based extractors.
2.  **Build** a `PropertyGraphIndex` to store these entities and relationships.
3.  **Query** using versatile retrievers that can perform keyword search, vector search, or even generate database queries.
4.  **Combine** multiple retrieval strategies for robust, comprehensive answers.

This approach is ideal for building complex, queryable knowledge bases from unstructured text, enabling sophisticated question-answering that understands relationships between entities.