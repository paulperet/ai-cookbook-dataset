# Building a Text-to-SQL System with Mistral AI, Neon, and LangChain

This guide walks you through building a Retrieval-Augmented Generation (RAG) system that converts natural language questions into executable SQL queries. By combining Mistral AI's language and embedding models with Neon's vector database capabilities, we create a system that understands your database schema and learns from example queries to generate accurate SQL.

## Prerequisites

Before you begin, ensure you have:
1. A **Mistral AI API Key** from the [Mistral AI console](https://console.mistral.ai/).
2. A **Neon Project** with a connection string from the [Neon Console](https://console.neon.tech/).
3. Python installed on your system.

## Step 1: Install Required Libraries

Open your terminal or notebook and install the necessary Python packages.

```bash
pip install langchain langchain-mistralai langchain-postgres sqlalchemy
```

## Step 2: Configure API and Database Credentials

Set your API keys and database connection string. Replace the placeholder values with your actual credentials.

```python
MISTRAL_API_KEY = "your-mistral-api-key"
NEON_CONNECTION_STRING = "your-neon-connection-string"
```

## Step 3: Set Up the Database Connection and Vector Store

We'll use SQLAlchemy to connect to Neon and LangChain's `PGVector` to store and query embeddings.

```python
import sqlalchemy
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

# 1. Create a database engine
engine = sqlalchemy.create_engine(
    url=NEON_CONNECTION_STRING,
    pool_pre_ping=True,
    pool_recycle=300
)

# 2. Initialize the Mistral embedding model
embeds_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)

# 3. Create a vector store in Postgres
vector_store = PGVector(
    embeddings=embeds_model,
    connection=engine,
    use_jsonb=True,
    collection_name="text-to-sql-context"
)
```

## Step 4: Prepare the Knowledge Base Data

We'll use the Northwind sample database. The knowledge base consists of two parts:
1. **Database Schema (DDL Statements):** Table definitions and relationships.
2. **Example Query Pairs:** Natural language questions paired with their corresponding SQL queries.

First, download the required data files.

```python
import os
import requests

# Create a data directory
os.makedirs("data", exist_ok=True)

# URLs for the sample data
repo_url = "https://raw.githubusercontent.com/neondatabase/mistral-neon-text-to-sql/main/data/"
files = ["northwind-schema.sql", "northwind-queries.jsonl"]

# Download each file
for filename in files:
    response = requests.get(repo_url + filename)
    with open(f"data/{filename}", "w") as f:
        f.write(response.text)
    print(f"Downloaded: {filename}")
```

### 4.1 Generate Embeddings for the Database Schema

Parse the SQL schema file and store each DDL statement as a document in the vector store.

```python
# Read and parse the schema file
all_statements = []
with open("data/northwind-schema.sql", "r") as f:
    current_statement = ""
    for line in f:
        # Skip empty lines and comments
        if line.strip() == "" or line.startswith("--"):
            continue
        current_statement += line
        if ";" in current_statement:
            all_statements.append(current_statement.strip())
            current_statement = ""

# Filter for CREATE and ALTER statements
ddl_statements = [stmt for stmt in all_statements if stmt.startswith(("CREATE", "ALTER"))]

# Create Document objects
schema_docs = [
    Document(
        page_content=stmt,
        metadata={"id": f"ddl-{i}", "topic": "ddl"}
    )
    for i, stmt in enumerate(ddl_statements)
]

# Add to vector store
vector_store.add_documents(
    schema_docs,
    ids=[doc.metadata["id"] for doc in schema_docs]
)
print(f"Added {len(schema_docs)} schema documents to the vector store.")
```

### 4.2 Generate Embeddings for Example Queries

Load the example question-query pairs and store them as documents.

```python
import json

# Load the example queries
example_docs = []
with open("data/northwind-queries.jsonl", "r") as f:
    for i, line in enumerate(f):
        example_docs.append(
            Document(
                page_content=line.strip(),
                metadata={"id": f"query-{i}", "topic": "query"}
            )
        )

# Add to vector store
vector_store.add_documents(
    example_docs,
    ids=[doc.metadata["id"] for doc in example_docs]
)
print(f"Added {len(example_docs)} example query documents to the vector store.")
```

### 4.3 Create the Northwind Tables in Neon

Execute the full schema script to create the actual tables in your Neon database. This allows us to run the generated SQL queries later.

```python
# Execute the DDL script to create the database tables
with engine.connect() as conn:
    with open("data/northwind-schema.sql") as f:
        conn.execute(sqlalchemy.text(f.read()))
    conn.commit()
print("Northwind database tables created successfully.")
```

## Step 5: Retrieve Relevant Context for a User Query

Now that our knowledge base is ready, we can retrieve relevant information for any natural language question.

Define your user question.

```python
question = "Find the employee who has processed the most orders and display their full name and the number of orders they have processed?"
```

### 5.1 Retrieve Relevant Schema Information

Search the vector store for DDL statements similar to the user's question.

```python
relevant_ddl = vector_store.similarity_search(
    query=question,
    k=5,  # Retrieve top 5 matches
    filter={"topic": {"$eq": "ddl"}}
)

print(f"Retrieved {len(relevant_ddl)} relevant DDL statements.")
```

### 5.2 Retrieve Similar Example Queries

Retrieve example question-query pairs that are semantically similar to the user's question. This provides few-shot examples to guide the LLM.

```python
similar_queries = vector_store.similarity_search(
    query=question,
    k=3,  # Retrieve top 3 matches
    filter={"topic": {"$eq": "query"}}
)

print(f"Retrieved {len(similar_queries)} similar example queries.")
```

## Step 6: Construct the LLM Prompt

We'll build a structured prompt that includes the retrieved schema, examples, and the user's question. The prompt instructs the LLM to reason step-by-step before outputting the SQL.

```python
prompt_template = """
You are an AI assistant that converts natural language questions into SQL queries. To do this, you will be provided with three key pieces of information:

1. Some DDL statements describing tables, columns and indexes in the database:
<schema>
{SCHEMA}
</schema>

2. Some example pairs demonstrating how to convert natural language text into a corresponding SQL query for this schema:
<examples>
{EXAMPLES}
</examples>

3. The actual natural language question to convert into an SQL query:
<question>
{QUESTION}
</question>

Follow the instructions below:
1. Your task is to generate an SQL query that will retrieve the data needed to answer the question, based on the database schema.
2. First, carefully study the provided schema and examples to understand the structure of the database and how the examples map natural language to SQL for this schema.
3. Your answer should have two parts:
- Inside <scratchpad> XML tag, write out step-by-step reasoning to explain how you are generating the query based on the schema, example, and question.
- Then, inside <sql> XML tag, output your generated SQL.
"""

# Format the retrieved schema and examples
schema_text = ""
for doc in relevant_ddl:
    schema_text += doc.page_content + "\n\n"

examples_text = ""
for doc in similar_queries:
    pair = json.loads(doc.page_content)
    examples_text += f"Question: {pair['question']}\n"
    examples_text += f"SQL: {pair['query']}\n\n"

# Create the final prompt
final_prompt = prompt_template.format(
    QUESTION=question,
    SCHEMA=schema_text,
    EXAMPLES=examples_text
)
```

## Step 7: Generate the SQL Query with Mistral AI

Use Mistral AI's chat model to process the prompt and generate the SQL query.

```python
import re
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage

# Initialize the chat model
chat_model = ChatMistralAI(api_key=MISTRAL_API_KEY)

# Generate the response
response = chat_model.invoke([
    HumanMessage(content=final_prompt)
])

# Extract the SQL from the response
sql_match = re.search(r"<sql>(.*?)</sql>", response.content, re.DOTALL)
if sql_match:
    sql_query = sql_match.group(1).strip()
    print("Generated SQL Query:")
    print(sql_query)
else:
    print("Could not extract SQL from the model's response.")
    print("Full response:", response.content)
```

## Step 8: Execute and Verify the Generated SQL

Run the generated SQL query against your Neon database to verify it works correctly.

```python
from sqlalchemy import text

try:
    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        print("\nQuery Results:")
        for row in result:
            print(dict(row._mapping))
except Exception as e:
    print(f"Error executing SQL: {e}")
```

## Production Considerations

While this tutorial provides a working prototype, consider these points for a production system:

1. **SQL Validation & Safety:** Always validate generated SQL, especially for `DELETE`, `UPDATE`, or `DROP` operations. Implement safeguards against SQL injection.
2. **Performance Monitoring:** Track the accuracy and latency of generated queries. Retrain or update embeddings as your database schema evolves.
3. **Enhanced Metadata:** Incorporate data lineage, query logs, or dashboard context to improve retrieval quality.
4. **Advanced Retrieval:** Techniques like Hypothetical Document Embeddings (HyDE) can improve the relevance of retrieved snippets for complex queries.

## Appendix: Data Sources

The Northwind database schema and sample queries were sourced from:
- [Northwind Psql](https://github.com/pthom/northwind_psql/blob/master/northwind.sql)
- [SQL Northwind Exercises](https://github.com/eirkostop/SQL-Northwind-exercises)

## Conclusion

You've successfully built a RAG-powered text-to-SQL system using Mistral AI for embeddings and generation, Neon as a vector database, and LangChain for orchestration. This system retrieves relevant database schema and example queries to help an LLM generate accurate, executable SQL from natural language questions.