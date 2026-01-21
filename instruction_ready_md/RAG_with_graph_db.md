# Building a Product Recommendation Chatbot with Neo4j and RAG

This guide demonstrates how to build a Retrieval Augmented Generation (RAG) system using a graph database (Neo4j) and Large Language Models (LLMs). You will create a product recommendation chatbot that intelligently queries a graph of Amazon products.

## Why Use RAG with a Graph Database?

**RAG** allows you to ground LLM responses in your own data, reducing hallucinations and providing up-to-date, relevant information.

**Graph Databases** like Neo4j excel at managing connected data. They are ideal when relationships between data points are crucial, such as for:
- Navigating deep hierarchies.
- Discovering hidden connections.
- Building recommendation systems, CRM tools, or customer behavior analyzers.

This tutorial combines both to create a powerful product recommendation engine.

## Prerequisites & Setup

Before you begin, ensure you have:
1. An OpenAI API key.
2. A running Neo4j instance (local or AuraDB).
3. Python installed on your machine.

### Step 1: Install Required Libraries
Install the necessary Python packages.

```bash
pip install langchain openai neo4j python-dotenv pandas
```

### Step 2: Import Libraries and Set Environment Variables
Import the libraries and load your OpenAI API key.

```python
import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Alternatively, set the API key directly
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Step 3: Load and Explore the Dataset

You will use a pre-processed JSON dataset containing Amazon products and their relationships.

### Load the Dataset
Load the dataset from a JSON file.

```python
file_path = 'data/amazon_product_kg.json'

with open(file_path, 'r') as file:
    jsonData = json.load(file)

# Convert to DataFrame for easy viewing
df = pd.read_json(file_path)
print(df.head())
```

## Step 4: Connect to the Neo4j Database

Establish a connection to your Neo4j instance.

```python
# Replace with your Neo4j credentials
url = "bolt://localhost:7687"
username = "neo4j"
password = "your-password-here"

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)
```

## Step 5: Import Data into the Graph

Define a function to sanitize text and create nodes and relationships in Neo4j.

```python
def sanitize(text):
    """Clean text for safe Cypher query insertion."""
    return str(text).replace("'", "").replace('"', '').replace('{', '').replace('}', '')

# Iterate through each JSON object and add it to the graph
for i, obj in enumerate(jsonData, start=1):
    print(f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}")

    query = f'''
        MERGE (product:Product {{id: {obj['product_id']}}})
        ON CREATE SET product.name = "{sanitize(obj['product'])}",
                       product.title = "{sanitize(obj['TITLE'])}",
                       product.bullet_points = "{sanitize(obj['BULLET_POINTS'])}",
                       product.size = {sanitize(obj['PRODUCT_LENGTH'])}

        MERGE (entity:{obj['entity_type']} {{value: "{sanitize(obj['entity_value'])}"}})

        MERGE (product)-[:{obj['relationship']}]->(entity)
        '''
    graph.query(query)
```

## Step 6: Create Vector Indexes for Semantic Search

To enable semantic search, create vector indexes on product and entity properties using OpenAI embeddings.

```python
embeddings_model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=embeddings_model)

# Create a vector index for Products
product_vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    url=url,
    username=username,
    password=password,
    index_name='products',
    node_label="Product",
    text_node_properties=['name', 'title'],
    embedding_node_property='embedding',
)

# Create vector indexes for all entity types
def embed_entities(entity_type):
    Neo4jVector.from_existing_graph(
        embeddings,
        url=url,
        username=username,
        password=password,
        index_name=entity_type,
        node_label=entity_type,
        text_node_properties=['value'],
        embedding_node_property='embedding',
    )

# Get unique entity types from the dataset
entities_list = df['entity_type'].unique()
for entity_type in entities_list:
    embed_entities(entity_type)
```

## Step 7: Extract Query Intent with an LLM

Instead of having an LLM generate Cypher directly (which is error-prone), you will use it to identify relevant entities from the user's query. These entities will then be used in templated Cypher queries.

First, define the schema of your graph.

```python
entity_types = {
    "product": "Item detailed type, e.g., 'high waist pants', 'outdoor plant pot'",
    "category": "Item category, e.g., 'home decoration', 'women clothing'",
    "characteristic": "Item characteristics, e.g., 'waterproof', 'adhesive'",
    "measurement": "Dimensions of the item",
    "brand": "Brand of the item",
    "color": "Color of the item",
    "age_group": "Target age group: 'babies', 'children', 'teenagers', 'adults'"
}

relation_types = {
    "hasCategory": "item is of this category",
    "hasCharacteristic": "item has this characteristic",
    "hasMeasurement": "item is of this measurement",
    "hasBrand": "item is of this brand",
    "hasColor": "item is of this color",
    "isFor": "item is for this age_group"
}

# Map entity types to their corresponding relationship types
entity_relationship_match = {
    "category": "hasCategory",
    "characteristic": "hasCharacteristic",
    "measurement": "hasMeasurement",
    "brand": "hasBrand",
    "color": "hasColor",
    "age_group": "isFor"
}
```

Next, create a system prompt that instructs the LLM to extract these entities from a user query.

```python
system_prompt = f'''
You are a helpful agent designed to fetch information from a graph database.

The graph database links products to the following entity types:
{json.dumps(entity_types, indent=2)}

Each link has one of the following relationships:
{json.dumps(relation_types, indent=2)}

Your task is to analyze the user's prompt and extract any mentioned entities that match the defined types.

Return a JSON object where each key is an exact match for one of the entity types above, and the value is the relevant text from the user query.

Example:
User: "Which blue clothing items are suitable for adults?"
Output: {{"color": "blue", "category": "clothing", "age_group": "adults"}}

If no relevant entities are found, return an empty JSON object: {{}}.
'''
```

Now, define a function that uses this prompt with the OpenAI API.

```python
def define_query(prompt, model="gpt-4o"):
    """Use an LLM to extract relevant entities from a user prompt."""
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Test the function
test_queries = [
    "Which pink items are suitable for children?",
    "Help me find gardening gear that is waterproof",
    "I'm looking for a bench with dimensions 100x50 for my living room"
]

for query in test_queries:
    print(f"Q: '{query}'")
    print(f"A: {define_query(query)}\n")
```

## Step 8: Generate and Execute Cypher Queries

With the extracted entities, you can now generate a Cypher query that uses vector similarity to find matching products.

First, create helper functions for generating embeddings and building the query.

```python
def create_embedding(text):
    """Generate an embedding for a given text."""
    result = client.embeddings.create(model=embeddings_model, input=text)
    return result.data[0].embedding

def create_query(extracted_entities_json, threshold=0.81):
    """
    Generate a Cypher query from extracted entities.
    The query finds products linked to entities with similar embeddings.
    """
    query_data = json.loads(extracted_entities_json)

    # Build the WITH clause for embeddings
    embeddings_clause = []
    for key in query_data:
        if key != 'product':
            embeddings_clause.append(f"${key}Embedding AS {key}Embedding")
    query = "WITH " + ",\n".join(embeddings_clause)

    # Build the MATCH clause for products and entities
    query += "\nMATCH (p:Product)\nMATCH "
    match_clauses = []
    for key in query_data:
        if key != 'product':
            relationship = entity_relationship_match[key]
            match_clauses.append(f"(p)-[:{relationship}]->({key}Var:{key})")
    query += ",\n".join(match_clauses)

    # Build the WHERE clause for similarity
    query += "\nWHERE "
    similarity_conditions = []
    for key in query_data:
        if key != 'product':
            similarity_conditions.append(f"gds.similarity.cosine({key}Var.embedding, ${key}Embedding) > {threshold}")
    query += " AND ".join(similarity_conditions)

    query += "\nRETURN p"
    return query

def query_graph(extracted_entities_json):
    """Execute the generated Cypher query and return matching products."""
    # Generate the query
    cypher_query = create_query(extracted_entities_json)

    # Prepare parameters: embeddings for each entity
    query_data = json.loads(extracted_entities_json)
    params = {}
    for key, value in query_data.items():
        if key != 'product':
            params[f"{key}Embedding"] = create_embedding(value)

    # Execute the query
    result = graph.query(cypher_query, params=params)
    return result

# Test the query generation and execution
example_response = '{"category": "clothes", "color": "blue", "age_group": "adults"}'
result = query_graph(example_response)

print(f"Found {len(result)} matching product(s):\n")
for record in result:
    product = record['p']
    print(f"{product['name']} (ID: {product['id']})")
```

## Step 9: Find Similar Items Using Graph Relationships

A key advantage of a graph database is the ability to traverse relationships to find similar items. Define a function that finds products sharing categories or a minimum number of common characteristics.

```python
def query_similar_items(product_id, relationships_threshold=3):
    """Find products similar to a given product ID."""
    similar_items = []

    # 1. Find items in the same category with at least one other entity in common
    query_category = '''
        MATCH (p:Product {id: $product_id})-[:hasCategory]->(c:category)
        MATCH (p)-->(entity)
        WHERE NOT entity:category
        MATCH (n:Product)-[:hasCategory]->(c)
        MATCH (n)-->(commonEntity)
        WHERE commonEntity = entity AND p.id <> n.id
        RETURN DISTINCT n;
    '''
    result_category = graph.query(query_category, params={"product_id": int(product_id)})

    # 2. Find items with at least N entities in common
    query_common_entities = '''
        MATCH (p:Product {id: $product_id})-->(entity),
              (n:Product)-->(entity)
        WHERE p.id <> n.id
        WITH n, COUNT(DISTINCT entity) AS commonEntities
        WHERE commonEntities >= $threshold
        RETURN n;
    '''
    result_common_entities = graph.query(
        query_common_entities,
        params={"product_id": int(product_id), "threshold": relationships_threshold}
    )

    # Combine results, avoiding duplicates
    for record in result_category:
        product = record['n']
        similar_items.append({"id": product['id'], "name": product['name']})

    for record in result_common_entities:
        product = record['n']
        if not any(item['id'] == product['id'] for item in similar_items):
            similar_items.append({"id": product['id'], "name": product['name']})

    return similar_items

# Test the function
test_product_ids = ['1519827', '2763742']
for pid in test_product_ids:
    print(f"Similar items for product #{pid}:")
    similar = query_similar_items(pid)
    for item in similar:
        print(f"  - {item['name']} (ID: {item['id']})")
    print()
```

## Step 10: Implement a Fallback Similarity Search

If the LLM cannot extract specific entities from the prompt, you can fall back to a generic similarity search against product names and titles.

```python
def similarity_search(prompt, threshold=0.8):
    """Perform a semantic similarity search on product embeddings."""
    embedding = create_embedding(prompt)
    query = '''
        WITH $embedding AS inputEmbedding
        MATCH (p:Product)
        WHERE gds.similarity.cosine(inputEmbedding, p.embedding) > $threshold
        RETURN p
    '''
    result = graph.query(query, params={'embedding': embedding, 'threshold': threshold})

    matches = []
    for record in result:
        product = record['p']
        matches.append({"id": product['id'], "name": product['name']})
    return matches

# Test the fallback search
test_prompt = "I'm looking for nice curtains"
print(similarity_search(test_prompt))
```

## Step 11: Build the Final RAG Pipeline

Combine all components into a unified function that:
1. Attempts to extract entities and query the graph.
2. Falls back to a similarity search if no entities are found.
3. Can optionally find similar items for the results.

```python
def recommend_products(user_prompt, find_similar=False, similar_threshold=3):
    """
    Main recommendation function.
    Args:
        user_prompt: The user's query.
        find_similar: If True, also find similar items for each match.
        similar_threshold: Minimum common entities for similarity.
    """
    # Step 1: Extract entities
    extracted_json = define_query(user_prompt)
    extracted_data = json.loads(extracted_json)

    matches = []

    if extracted_data:
        # Step 2: Query graph with extracted entities
        print("Querying graph with extracted entities...")
        graph_result = query_graph(extracted_json)
        for record in graph_result:
            product = record['p']
            match = {"id": product['id'], "name": product['name']}
            if find_similar:
                match["similar_items"] = query_similar_items(product['id'], similar_threshold)
            matches.append(match)
    else:
        # Step 3: Fallback to similarity search
        print("No entities extracted. Performing similarity search...")
        matches = similarity_search(user_prompt)

    return matches

# Test the complete pipeline
user_query = "Find me a large, waterproof backpack for hiking"
recommendations = recommend_products(user_query, find_similar=True)

print(f"\nRecommendations for: '{user_query}'")
for rec in recommendations:
    print(f"\n- {rec['name']} (ID: {rec['id']})")
    if 'similar_items' in rec:
        print("  Similar items:")
        for sim in rec['similar_items']:
            print(f"    * {sim['name']} (ID: {sim['id']})")
```

## Summary and Next Steps

You have successfully built a RAG system with Neo4j that:
1. **Stores** product data in a graph with meaningful relationships.
2. **Indexes** text properties with vector embeddings for semantic search.
3. **Uses an LLM** to intelligently extract query intent.
4. **Generates and executes** precise Cypher queries using vector similarity.
5. **Leverages graph traversals** to find similar items.
6. **Provides a fallback** semantic search when entity extraction fails.

### Potential Enhancements
- **Add a Chat Interface:** Integrate with a framework like Chainlit or Gradio for a conversational UI.
- **Improve Prompt Engineering:** Refine the system prompt for better entity extraction.
- **Hybrid Search:** Combine vector similarity with keyword matching for more robust results.
- **Response Generation:** Use an LLM to generate natural language responses from the retrieved products.

This architecture provides a flexible foundation for building sophisticated recommendation systems and knowledge-aware chatbots using your own data.