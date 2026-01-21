# Building a Robust Text-to-SQL Agent with Automatic Error Correction

In this guide, you'll implement an intelligent agent that executes SQL queries with built-in error correction using the `smolagents` framework. Unlike traditional text-to-SQL pipelines that fail silently with incorrect queries, this agent critically evaluates outputs and iteratively corrects mistakes.

## Prerequisites

First, install the required packages:

```bash
pip install sqlalchemy smolagents
```

## Step 1: Set Up the Database

Create an in-memory SQLite database with a sample table structure.

```python
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert,
    inspect,
    text,
)

# Create database engine and metadata
engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# Define the receipts table
table_name = "receipts"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("customer_name", String(16), primary_key=True),
    Column("price", Float),
    Column("tip", Float),
)

# Create all tables
metadata_obj.create_all(engine)
```

## Step 2: Populate with Sample Data

Insert sample records to work with.

```python
rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
]

for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)
```

Verify the data was inserted correctly:

```python
with engine.connect() as con:
    rows = con.execute(text("SELECT * from receipts"))
    for row in rows:
        print(row)
```

## Step 3: Create the SQL Tool

Define a tool that allows the agent to execute SQL queries. The tool's docstring is crucial—it provides the LLM with schema information and usage instructions.

```python
from smolagents import tool

@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'receipts'. Its description is as follows:
        Columns:
        - receipt_id: INTEGER
        - customer_name: VARCHAR(16)
        - price: FLOAT
        - tip: FLOAT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output
```

## Step 4: Initialize the Agent

Create a `CodeAgent` that uses the SQL tool. This agent follows the ReAct framework, writing and executing code while iterating based on results.

```python
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel("meta-llama/Meta-Llama-3-8B-Instruct"),
)
```

## Step 5: Test with a Simple Query

Run your first query through the agent:

```python
agent.run("Can you give me the name of the client who got the most expensive receipt?")
```

The agent should correctly identify Woodrow Wilson as having the most expensive receipt.

## Step 6: Add Complexity with Table Joins

To demonstrate the agent's ability to handle more complex scenarios, create a second table for waiters.

```python
table_name = "waiters"
waiters = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("waiter_name", String(16), primary_key=True),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "waiter_name": "Corey Johnson"},
    {"receipt_id": 2, "waiter_name": "Michael Watts"},
    {"receipt_id": 3, "waiter_name": "Michael Watts"},
    {"receipt_id": 4, "waiter_name": "Margaret James"},
]

for row in rows:
    stmt = insert(waiters).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)
```

## Step 7: Update the Tool Description

Update the SQL tool's description to include both tables' schemas so the LLM can properly construct joins.

```python
updated_description = """Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
It can use the following tables:"""

inspector = inspect(engine)
for table in ["receipts", "waiters"]:
    columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table)]
    
    table_description = f"Table '{table}':\n"
    table_description += "Columns:\n" + "\n".join(
        [f"  - {name}: {col_type}" for name, col_type in columns_info]
    )
    updated_description += "\n\n" + table_description

sql_engine.description = updated_description
```

## Step 8: Use a More Powerful Model for Complex Queries

Switch to a more capable model for handling the increased complexity of multi-table queries.

```python
agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel("Qwen/Qwen2.5-72B-Instruct"),
)
```

## Step 9: Execute a Complex Query with Joins

Test the agent with a query that requires joining tables and aggregating data:

```python
agent.run("Which waiter got more total money from tips?")
```

The agent should correctly determine that Michael Watts received the most in total tips ($5.67) by joining the `receipts` and `waiters` tables and performing the necessary aggregation.

## Conclusion

You've successfully built a text-to-SQL agent that:
1. Executes SQL queries based on natural language prompts
2. Automatically corrects errors through iterative refinement
3. Handles complex operations like table joins and aggregations
4. Adapts to different LLM backends based on task complexity

This approach provides significant advantages over traditional text-to-SQL pipelines by incorporating critical evaluation and self-correction mechanisms, resulting in more reliable and accurate query execution.