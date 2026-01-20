# Building a Text-to-SQL conversion system with Mistral AI, Neon, and LangChain

Translating natural language queries into SQL statements is a powerful application of large language models (LLMs). While it's possible to ask an LLM directly to generate SQL based on a natural language prompt, it comes with limitations.

1. The LLM may generate SQL that is syntactically incorrect since the SQL dialect varies across relational databases.
2. The LLM doesn't have access to the full database schema, table and column names or indexes, which limits its ability to generate accurate/efficient queries. Passing in the full schema as input to the LLM everytime can get expensive.
3. Pretrained LLMs can't adapt to user feedback and evolving text queries.

### Finetuning

An alternative is to finetune the LLM on your specific text-to-SQL dataset, which might includes query logs from your database and other relevant context. While this approach can improve the LLM's ability to generate accurate SQL queries, it still has limitations adapting continuously. Finetuning can also be expensive which might limit how frequently you can update the model.

### RAG systems

LLMs are great at in-context learning, so by feeding them relevant information in the prompt, we can improve their outputs. This is the idea behind Retrieval Augmented Generation (RAG) systems, which combine information retrieval with LLMs to generate more informed and contextual responses to queries.

By retrieving relevant information from a knowledge base - database schemas, which tables to query, and previously generated SQL queries, we can leverage LLMs to generate SQL queries that are more accurate and efficient.

### RAG for text-to-sql

In this post, we'll walk through building a RAG system using [Mistral AI](https://mistral.ai/) for embeddings and language modeling, [Neon Postgres](https://neon.tech/) for the vector database. `Neon` is a fully managed serverless PostgreSQL database. It separates storage and compute to offer features such as instant branching and automatic scaling. With the `pgvector` extension, Neon can be used as a vector database to store text embeddings and query them.

We'll set up a sample database, generate and store embeddings for a knowledge-base about it, and then retrieve relevant snippets to answer a query. We use the popular [LangChain](https://www.langchain.com/) library to tie it all together.

Let's dive in!

## Setup and Dependencies

### Mistral AI API

Sign up at [Mistral AI](https://mistral.ai/) and navigate to the console. From the sidebar, go to the `API keys` section and create a new API key.

You'll need this key to access Mistral AI's embedding and language models. Set the variable below to it.

```python
MISTRAL_API_KEY = "your-mistral-api-key"
```

### Neon Database

Sign up at [Neon](https://neon.tech/) if you don't already have an account. Your Neon project comes with a ready-to-use Postgres database named `neondb` which we'll use in this notebook.

Log in to the Neon Console and navigate to the Connection Details section to find your database connection string. It should look similar to this:

```text
postgres://alex:AbC123dEf@ep-cool-darkness-123456.us-east-2.aws.neon.tech/dbname?sslmode=require
```

Set the variable below to the Neon connection string.

```python
NEON_CONNECTION_STRING = "your-neon-connection-string"
```

### Python Libraries

Install the necessary libraries to create the RAG system.

```python
%pip install langchain langchain-mistralai langchain-postgres
```

`langchain-postgres` provides a `vectorstore` module that allows us to store and query embeddings in a Postgres database with `pgvector` installed. While, we need `langchain-mistralai` to interact with `Mistral` models.

### Preparing the Data

For our example, we'll leverage the commonly used Northwind sample dataset. It models a fictional trading company called `Northwind Traders` that sells products to customers. It has tables representing entities such as `Customers`, `Orders`, `Products`, and `Employees`, interconnected through relationships, allowing users to query and analyze data related to sales, inventory and other business operations.

We want to provide two pieces of information as context when calling the Mistral LLM:

- Relevant tables/index information from the Northwind database schema
- Some sample (text-question, sql query) pairs for the LLM to learn from.

We will set up retrieval by leveraging a vector database to store the schema and the sample (text, sql) pairs. We create embeddings using the `mistral-embed` LLM model for each piece of information and at query time, retrieve the relevant snippets by comparing the query embedding with the stored embeddings.

We'll use the `langchain-postgres` library to store embeddings of the data in the database.

```python
import sqlalchemy

# Connect to the database
engine = sqlalchemy.create_engine(
    url=NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
)
```

```python
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

embeds_model = MistralAIEmbeddings(model="mistral-embed", api_key=MISTRAL_API_KEY)
vector_store = PGVector(
    embeddings=embeds_model,
    connection=engine,
    use_jsonb=True,
    collection_name="text-to-sql-context",
)
```

Next, we generate embeddings for the Northwind schema and sample queries.

The `add_documents` method on a langchain vector store, like `PGVector` here uses the specified embeddings model to generate embeddings for the input text and stores them in the database.

**NOTE**: If working in Colab, download the database setup and sample query scripts by running this

```python
# import os
# import requests

# repo_url = "https://raw.githubusercontent.com/neondatabase/mistral-neon-text-to-sql/main/data/"
# fnames = ["northwind-schema.sql", "northwind-queries.jsonl"]

# os.mkdir("data")
# for fname in fnames:
#     response = requests.get(repo_url + fname)
#     with open(f"data/{fname}", "w") as file:
#         file.write(response.text)
```

```python
# DDL statements to create the Northwind database

_all_stmts = []
with open("data/northwind-schema.sql", "r") as f:
    stmt = ""
    for line in f:
        if line.strip() == "" or line.startswith("--"):
            continue
        else:
            stmt += line
            if ";" in stmt:
                _all_stmts.append(stmt.strip())
                stmt = ""

ddl_stmts = [x for x in _all_stmts if x.startswith(("CREATE", "ALTER"))]

docs = [
    Document(page_content=stmt, metadata={"id": f"ddl-{i}", "topic": "ddl"})
    for i, stmt in enumerate(ddl_stmts)
]
vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
```

```python
# Sample question-query pairs

with open("data/northwind-queries.jsonl", "r") as f:
    docs = [
        Document(
            page_content=pair,
            metadata={"id": f"query-{i}", "topic": "query"},
        )
        for i, pair in enumerate(f)
    ]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
```

We will also create the Northwind tables in our Neon database, so we can execute the LLM output and have a working natural-language to query results pipeline.

```python
# run the DDL script to create the database
with engine.connect() as conn:
    with open("data/northwind-schema.sql") as f:
        conn.execute(sqlalchemy.text(f.read()))
    conn.commit()
```

### Retrieving Relevant Information

With our knowledge base set up, we can now retrieve relevant information for a given query.

Consider a user asking the query below.

```python
question = "Find the employee who has processed the most orders and display their full name and the number of orders they have processed?"
```

We use the `similarity search` method on the vector store to retrieve the most similar snippets to the query.

```python
relevant_ddl_stmts = vector_store.similarity_search(
    query=question, k=5, filter={"topic": {"$eq": "ddl"}}
)

# relevant_ddl_stmts
```

We also fetch some similar queries from our example corpus to add to the LLM prompt. `Few shot` prompting by providing examples of the text-to-sql conversion task in this manner helps the LLM generate more relevant output.

```python
similar_queries = vector_store.similarity_search(
    query=question, k=3, filter={"topic": {"$eq": "query"}}
)

# similar_queries
```

### Generating the SQL output

Finally, we'll use Mistral AI's chat model to generate a SQL statement based on the retrieved context.

We first construct the prompt we pass to the Mistral AI model. The prompt includes the query, the retrieved schema snippets, and some similar queries from the corpus.

```python
import json

prompt = """
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

schema = ""
for stmt in relevant_ddl_stmts:
    schema += stmt.page_content + "\n\n"

examples = ""
for stmt in similar_queries:
    text_sql_pair = json.loads(stmt.page_content)
    examples += "Question: " + text_sql_pair["question"] + "\n"
    examples += "SQL: " + text_sql_pair["query"] + "\n\n"
```

Prompting the LLM to think step by step improves the quality of the generated output. Hence, we instruct the LLM to output its reasoning and the SQL query in separate blocks in the output text.

We then call the Mistral AI model to generate the SQL statement.

```python
import re
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage

chat_model = ChatMistralAI(api_key=MISTRAL_API_KEY)
response = chat_model.invoke(
    [
        HumanMessage(
            content=prompt.format(QUESTION=question, SCHEMA=schema, EXAMPLES=examples)
        )
    ]
)

sql_query = re.search(r"<sql>(.*?)</sql>", response.content, re.DOTALL).group(1)
print(sql_query)
```

We extract the SQL statement from the Mistral AI model's output and execute it on the Neon database to verify if it is valid.

```python
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text(sql_query))
    for row in result:
        print(row._mapping)
```

## Conclusion

Thus, we have a working text-question-to-SQL query system by leveraging the `Mistral AI` API for both chat and embedding models, and `Neon` as the vector database.

To use it in production, there are some other considerations to keep in mind:

1. Validate the generated SQL query, especially for destructive operations like `DELETE` and `UPDATE` before executing them. Since the text input comes from a user, it might also cause SQL injection attacks by prompting the system with malicious input.

2. Monitor the system's performance and accuracy over time. We might need to update the LLM model used and the knowledge base embeddings as the data evolves.

3. Better metadata. While similar examples and database schema are useful, information like data lineage and dashboard logs can add more context to the LLM API calls.

4. Improving retrieval. For complex queries, we might need to increase the schema information passed to the LLM model. Further, our similarity search heuristic is pretty naive in that we are matching text queries to SQL statements. Using techniques like HyDE (Hypothetical Document Expansion) can improve the quality of the retrieved snippets.

## Appendix

We fetched the Northwind database setup script and some sample queries from the following helpful repositories:

- [Northwind Psql](https://github.com/pthom/northwind_psql/blob/master/northwind.sql)
- [Sample queries](https://github.com/eirkostop/SQL-Northwind-exercises)

```python

```