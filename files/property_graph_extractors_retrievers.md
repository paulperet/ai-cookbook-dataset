# Extractors and Retrievers in Property Graph

In this notebook, we will explore how to define extractors and retrievers for the PropertyGraph Index.

A property graph is a structured collection of labeled nodes (such as entity categories and text labels) with properties (metadata), interconnected by relationships to form structured paths (triplets).

In LlamaIndex, the PropertyGraphIndex plays a crucial role in:

•	Constructing a graph

•	Querying a graph

## Building and Using PropertyGraph

```python
from llama_index.core import PropertyGraphIndex

# create
index = PropertyGraphIndex.from_documents(
    documents,
)

# use
retriever = index.as_retriever(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
nodes = retriever.retrieve("<QUERY>")

query_engine = index.as_query_engine(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
response = query_engine.query("<QUERY>")

# save and load
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import StorageContext, load_index_from_storage

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

# loading from existing graph store (and optional vector store)
# load from existing graph/vector store
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store, vector_store=vector_store, llm=llm
)
```

Property graph construction involves executing a series of knowledge graph extractors on each chunk, and attaching entities and relations as metadata to each node. 

You can use as many extractors as needed, and all will be applied.

```python
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[extractor1, extractor2, ...],
)

# insert additional documents / nodes
index.insert(document)
index.insert_nodes(nodes)
```

If not provided, the defaults are SimpleLLMPathExtractor and ImplicitPathExtractor.

```python
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,
    num_workers=4,
    show_progress=False,
)
```

### SimpleLLMPathExtractor

Use an LLM to extract short statements and parse single-hop paths in the format (entity1, relation, entity2).

If desired, you can also customize both the prompt and the function used for parsing the paths.

Here’s a straightforward (though simplistic) example:

```python
prompt = (
    "Some text is provided below. Given the text, extract up to "
    "{max_paths_per_chunk} "
    "knowledge triples in the form of `subject,predicate,object` on each line. Avoid stopwords.\n"
)


def parse_fn(response_str: str) -> List[Tuple[str, str, str]]:
    lines = response_str.split("\n")
    triples = [line.split(",") for line in lines]
    return triples


kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    extract_prompt=prompt,
    parse_fn=parse_fn,
)
```

### ImplicitPathExtractor

Extract paths using the node.relationships attribute on each LlamaIndex node object.

This extraction process does not require an LLM or embedding model, as it simply parses properties that already exist on the node objects.

```python
from llama_index.core.indices.property_graph import ImplicitPathExtractor

kg_extractor = ImplicitPathExtractor()
```

### SchemaLLMPathExtractor

Extract paths by adhering to a strict schema that specifies allowed entities, relationships, and the connections between them.

Using Pydantic, structured outputs from LLMs, and some intelligent validation, we can dynamically define a schema and verify the extractions for each path.(triplet)

```python
from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# recommended uppercase, underscore separated
entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["PART_OF", "HAS", "WORKED_AT"]
# schema = {
#     "PERSON": ["PART_OF", "HAS", "WORKED_AT"],
#     "PLACE": ["PART_OF", "HAS"],
#     "ORGANIZATION": ["WORKED_AT"],
# }

schema = [
    ("PLACE", "HAS", "PERSON"),
    ("PERSON", "PART_OF", "PLACE"),
    ("PERSON", "WORKED_AT", "ORGANIZATION")
]

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=schema,
    strict=True,  # if false, will allow triples outside of the schema
    num_workers=4,
    max_paths_per_chunk=10,
    show_progres=False,
)
```

## Retrieval and Querying

Labeled property graphs offer various querying methods to retrieve nodes and paths. In LlamaIndex, we have the ability to simultaneously combine multiple node retrieval techniques!

```python
# create a retriever
retriever = index.as_retriever(sub_retrievers=[retriever1, retriever2, ...])

# create a query engine
query_engine = index.as_query_engine(
    sub_retrievers=[retriever1, retriever2, ...]
)
```

If no sub-retrievers are specified, the default retrievers used are the LLMSynonymRetriever and VectorContextRetriever (if embeddings are enabled).

Currently, the following retrievers are included:

•	LLMSynonymRetriever: Retrieves nodes based on keywords and synonyms generated by an LLM.

•	VectorContextRetriever: Retrieves nodes based on embedded graph nodes.

•	TextToCypherRetriever: Directs the LLM to generate Cypher queries based on the schema of the property graph.

•	CypherTemplateRetriever: Utilizes a Cypher template with parameters inferred by the LLM.

•	CustomPGRetriever: Easily subclassed to implement custom retrieval logic.

```python
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)

sub_retrievers = [
    VectorContextRetriever(index.property_graph_store, ...),
    LLMSynonymRetriever(index.property_graph_store, ...),
]

retriever = PGRetriever(sub_retrievers=sub_retrievers)

nodes = retriever.retrieve("<query>")
```

### LLMSynonymRetriever

This retriever takes the input query and attempts to generate relevant keywords and synonyms. These are used to retrieve nodes and consequently, the paths connected to those nodes.

Explicitly declaring this retriever in your configuration allows for the customization of several options.

```python
from llama_index.core.indices.property_graph import LLMSynonymRetriever

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


def parse_fn(self, output: str) -> list[str]:
    matches = output.strip().split("^")

    # capitalize to normalize with ingestion
    return [x.strip().capitalize() for x in matches if x.strip()]


synonym_retriever = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    # include source chunk text with retrieved paths
    include_text=False,
    synonym_prompt=prompt,
    output_parsing_fn=parse_fn,
    max_keywords=10,
    # the depth of relations to follow after node retrieval
    path_depth=1,
)

retriever = index.as_retriever(sub_retrievers=[synonym_retriever])
```

## VectorContextRetriever

This retriever identifies nodes based on their vector similarity, subsequently fetching the paths connected to those nodes.

If your graph store natively supports vector capabilities, managing that graph store alone suffices for storage. However, if vector support is not inherent, you will need to supplement the graph store with a vector store. By default, this setup uses the in-memory SimpleVectorStore.

```python
from llama_index.core.indices.property_graph import VectorContextRetriever

vector_retriever = VectorContextRetriever(
    index.property_graph_store,
    # only needed when the graph store doesn't support vector queries
    # vector_store=index.vector_store,
    embed_model=embed_model,
    # include source chunk text with retrieved paths
    include_text=False,
    # the number of nodes to fetch
    similarity_top_k=2,
    # the depth of relations to follow after node retrieval
    path_depth=1,
    # can provide any other kwargs for the VectorStoreQuery class
    ...,
)

retriever = index.as_retriever(sub_retrievers=[vector_retriever])
```

### TextToCypherRetriever

This retriever utilizes a graph store schema, your query, and a prompt template for text-to-cypher conversion to generate and execute a Cypher query.

Note: Since the SimplePropertyGraphStore is not a full-fledged graph database, it does not support Cypher queries.

To inspect the schema, you can use the method: index.property_graph_store.get_schema_str().

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

DEFAULT_RESPONSE_TEMPLATE = (
    "Generated Cypher query:\n{query}\n\n" "Cypher Response:\n{response}"
)
DEFAULT_ALLOWED_FIELDS = ["text", "label", "type"]

DEFAULT_TEXT_TO_CYPHER_TEMPLATE = (
    index.property_graph_store.text_to_cypher_template,
)


cypher_retriever = TextToCypherRetriever(
    index.property_graph_store,
    # customize the LLM, defaults to Settings.llm
    llm=llm,
    # customize the text-to-cypher template.
    # Requires `schema` and `question` template args
    text_to_cypher_template=DEFAULT_TEXT_TO_CYPHER_TEMPLATE,
    # customize how the cypher result is inserted into
    # a text node. Requires `query` and `response` template args
    response_template=DEFAULT_RESPONSE_TEMPLATE,
    # an optional callable that can clean/verify generated cypher
    cypher_validator=None,
    # allowed fields in the resulting
    allowed_output_field=DEFAULT_ALLOWED_FIELDS,
)
```

### CypherTemplateRetriever

This is a more constrained version of the TextToCypherRetriever. Instead of allowing the LLM free rein to generate any Cypher statement, we can provide a Cypher template and have the LLM fill in the blanks.

```python
# NOTE: current v1 is needed
from pydantic.v1 import BaseModel, Field
from llama_index.core.indices.property_graph import CypherTemplateRetriever

# write a query with template params
cypher_query = """
MATCH (c:Chunk)-[:MENTIONS]->(o)
WHERE o.name IN $names
RETURN c.text, o.name, o.label;
"""


# create a pydantic class to represent the params for our query
# the class fields are directly used as params for running the cypher query
class TemplateParams(BaseModel):
    """Template params for a cypher query."""

    names: list[str] = Field(
        description="A list of entity names or keywords to use for lookup in a knowledge graph."
    )


template_retriever = CypherTemplateRetriever(
    index.property_graph_store, TemplateParams, cypher_query
)
```

```python

```