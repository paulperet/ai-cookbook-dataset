# Building a RAG Agent for a Neo4j Movie Database

This guide walks you through creating a Retrieval-Augmented Generation (RAG) agent that can answer natural language questions about a Neo4j movie database. You'll build a system that translates questions into Cypher queries, executes them, and generates natural language responses.

## Prerequisites

Before starting, ensure you have:
- A Neo4j database instance (local via Neo4j Desktop or cloud-based)
- A Mistral AI API key
- Python 3.7+

## Setup

First, install the required packages and configure your environment:

```bash
pip install mistralai neo4j
```

```python
import json
from neo4j import GraphDatabase
from mistralai import Mistral
from getpass import getpass

# Configure your API keys and database credentials
api_key = getpass("Type your Mistral AI API Key: ")
neo4j_password = getpass("Type your Neo4j password: ")
neo4j_user = getpass("Type your Neo4j username: ")
neo4j_uri = getpass("Type your Neo4j URL: ")

# Initialize the Mistral client
client = Mistral(api_key=api_key)
model = "mistral-large-latest"  # Specify your preferred model
```

## Step 1: Configure Database Connection

Set up a function to execute Cypher queries against your Neo4j database. This function will handle the connection lifecycle automatically:

```python
URI = neo4j_uri
AUTH = (neo4j_user, neo4j_password)

def run_cypher_query(cypher_query):
    """Execute a Cypher query and return the results."""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, _, _ = driver.execute_query(cypher_query, database_="neo4j")
    return records
```

## Step 2: Create a Text-to-Cypher Agent

This agent translates natural language questions into executable Cypher queries. It uses a structured prompt with examples to ensure accurate query generation:

```python
def generate_cypher_query(question):
    """Generate a Cypher query from a natural language question."""
    prompt = f"""You are a coding agent interacting with a Neo4j database with the following schema:

- Labels: "Movie", "Person"
- Relationships: "ACTED_IN", "DIRECTED", "FOLLOWS", "PRODUCED", "REVIEWED", "WROTE"

"Person" label has the following properties:
- born
- name

"Movie" label has the following properties:
- title
- released

You will be given a query in natural language and your role is to output a Cypher query whose output will contain the answer.
Your output will be in JSON format.

Examples:

input: When was the movie "The Matrix" released?
output: {{"result": "MATCH (n:Movie) WHERE n.title='The Matrix' RETURN n.released"}}

input: In which movies did Tom Hanks play?
output: {{"result": "MATCH (p:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN m.title AS movieTitle"}}

input: What movies did Steven Spielberg produce?
output: {{"result": "MATCH (p:Person {{name: 'Steven Spielberg'}})-[:PRODUCED]->(m:Movie) RETURN m.title AS movieTitle"}}

Here is the user question:
{question}
"""

    chat_response = client.chat.complete(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content
```

## Step 3: Create a Response Agent

This agent takes the original question, generated Cypher query, and query results to produce a natural language answer:

```python
def respond_to_query(question, cypher_code, query_output):
    """Generate a natural language response based on query results."""
    prompt = f"""You are a coding agent interacting with a Neo4j database with the following schema:

- Labels: "Movie", "Person"
- Relationships: "ACTED_IN", "DIRECTED", "FOLLOWS", "PRODUCED", "REVIEWED", "WROTE"

"Person" label has the following properties:
- born
- name

"Movie" label has the following properties:
- title
- released

The user asked the following question:
{question}

To answer the question, the following Cypher query was run on Neo4j:
{cypher_code}

The following output was obtained:
{query_output}

Based on all these elements, answer the initial user question.
Be straight to the point and concise in your answers.

Your answer:
"""

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content
```

## Step 4: Build the End-to-End Workflow

Combine all components into a single function that handles the complete question-answering pipeline:

```python
def neo4j_agent(question):
    """Complete RAG pipeline: question → Cypher query → execution → response."""
    # Step 1: Generate Cypher query
    cypher_code = json.loads(generate_cypher_query(question))['result']
    
    # Step 2: Execute query
    query_result = run_cypher_query(cypher_code)
    
    # Step 3: Generate response
    response = respond_to_query(question, cypher_code, query_result)
    
    # Display results
    print(f'Question:\n{question}\n')
    print(f'Generated Cypher Query:\n{cypher_code}\n')
    print(f'Response:\n{response}\n')
```

## Step 5: Test Your RAG Agent

Now let's test the agent with various questions to see it in action:

```python
# Test 1: Simple property query
neo4j_agent("When was Keanu Reeves born?")
```

**Output:**
```
Question:
When was Keanu Reeves born?

Generated Cypher Query:
MATCH (n:Person) WHERE n.name='Keanu Reeves' RETURN n.born

Response:
Keanu Reeves was born in 1964.
```

```python
# Test 2: Relationship query
neo4j_agent("What actors played in the movie The Matrix?")
```

**Output:**
```
Question:
What actors played in the movie The Matrix?

Generated Cypher Query:
MATCH (m:Movie {title: 'The Matrix'})<-[:ACTED_IN]-(p:Person) RETURN p.name AS actorName

Response:
Sure, based on the executed Cypher query and the output, the actors who played in the movie "The Matrix" are Keanu Reeves, Carrie-Anne Moss, Laurence Fishburne, Hugo Weaving, and Emil Eifrem.
```

```python
# Test 3: Complex query with sorting
neo4j_agent("List Tom Hanks movies and sort them by release date")
```

**Output:**
```
Question:
List Tom Hanks movies and sort them by release date

Generated Cypher Query:
MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN m.title AS movieTitle, m.released AS releaseDate ORDER BY m.released ASC

Response:
Tom Hanks has acted in several movies over the years. Here are Tom Hanks' movies sorted by their release date:

1. "Joe Versus the Volcano" (1990)
2. "A League of Their Own" (1992)
3. "Sleepless in Seattle" (1993)
4. "Apollo 13" (1995)
5. "That Thing You Do" (1996)
6. "You've Got Mail" (1998)
7. "The Green Mile" (1999)
8. "Cast Away" (2000)
9. "The Polar Express" (2004)
10. "The Da Vinci Code" (2006)
11. "Charlie Wilson's War" (2007)
12. "Cloud Atlas" (2012)
```

```python
# Test 4: Calculation query
neo4j_agent("We are in 2024, how old is Tom Hanks?")
```

**Output:**
```
Question:
We are in 2024, how old is Tom Hanks?

Generated Cypher Query:
MATCH (p:Person {name: 'Tom Hanks'}) RETURN 2024 - p.born AS age

Response:
According to the provided information, Tom Hanks is 68 years old in 2024.
```

```python
# Test 5: Complex pattern matching
neo4j_agent("Give me names of two actors that played together in two different films. Give me the names of the associated movies")
```

**Output:**
```
Question:
Give me names of two actors that played together in two different films. Give me the names of the associated movies

Generated Cypher Query:
MATCH (p1:Person)-[:ACTED_IN]->(m1:Movie)<-[:ACTED_IN]-(p2:Person)-[:ACTED_IN]->(m2:Movie) WHERE m1 <> m2 RETURN p1.name AS actor1, p2.name AS actor2, collect(DISTINCT m1.title) AS movies1, collect(DISTINCT m2.title) AS movies2 LIMIT 2

Response:
Sure, based on the output of the Cypher query, we can see that both 'Carrie-Anne Moss' and 'Laurence Fishburne' have acted together with 'Keanu Reeves' in multiple movies, specifically 'The Matrix Reloaded' and 'The Matrix Revolutions'. However, they have also acted together in 'The Matrix' which is also included in the list of movies for 'Keanu Reeves'.

To answer the user's question, we should focus on the unique pairs of actors who have acted together in two different films. In this case, 'Carrie-Anne Moss' and 'Keanu Reeves' are the only pair that meets this criteria, with 'The Matrix Reloaded' and 'The Matrix Revolutions' being the movies they acted together in.

Additionally, the list of movies for 'Keanu Reeves' includes other titles besides 'The Matrix' and 'The Matrix Reloaded', which shows his involvement in other projects as well.

So, the answer to the user's question is: 'Carrie-Anne Moss' and 'Keanu Reeves' are the two actors who have played together in two different films, and those films are 'The Matrix Reloaded' and 'The Matrix Revolutions'.
```

## Conclusion

You've successfully built a RAG agent that can answer complex questions about a Neo4j movie database. The system demonstrates:

1. **Natural language to Cypher translation** using structured prompting
2. **Secure database interaction** with proper connection management
3. **Context-aware response generation** that explains results clearly

This architecture can be extended to other domains by updating the schema description in the prompts and adding more example queries for better performance.