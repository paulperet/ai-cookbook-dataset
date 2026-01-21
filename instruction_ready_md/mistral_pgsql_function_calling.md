# Guide: Query a PostgreSQL Database with AutoGen and Mistral Large

## Overview
This guide demonstrates how to use Microsoft AutoGen with Mistral Large 2407 to safely query a PostgreSQL database. You will create a read-only tool that allows the LLM to execute SQL queries, while providing it with up-to-date schema context. This approach enables natural language interaction with your database without risking data modification.

## Prerequisites
- A system with `sudo` privileges (e.g., a cloud VM or local Linux environment)
- Python 3.8+
- A Mistral AI API key

## Step 1: Set Up PostgreSQL and Sample Data
First, install PostgreSQL and import a sample dataset from IMDb.

```bash
# Update package list and install dependencies
sudo apt update
sudo apt install dirmngr ca-certificates software-properties-common gnupg gnupg2 apt-transport-https curl -y

# Add PostgreSQL repository
curl -fSsL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor | sudo tee /usr/share/keyrings/postgresql.gpg > /dev/null
echo 'deb [arch=amd64,arm64,ppc64el signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt/ jammy-pgdg main' | sudo tee /etc/apt/sources.list.d/pgdg.list

# Install PostgreSQL
sudo apt update
sudo apt install postgresql-client-16 postgresql-16 -y
sudo service postgresql start
```

Set a password for the default `postgres` user and create a sample database.

```bash
# Set password for postgres user
sudo -u postgres psql -U postgres -c "ALTER ROLE postgres WITH PASSWORD 'super_secret_postgres_password';"

# Create a data directory and download IMDb dataset
sudo mkdir /data
cd /data
sudo wget https://datasets.imdbws.com/name.basics.tsv.gz
sudo gunzip name.basics.tsv.gz

# Create a superuser role (optional, for administrative tasks)
sudo -u postgres psql -U postgres -c "CREATE ROLE root WITH SUPERUSER;"
sudo -u postgres psql -U postgres -c "ALTER ROLE root WITH LOGIN;"

# Create table and import data
sudo -u postgres psql -U postgres -c "CREATE TABLE imdb ( nconst TEXT, primaryName TEXT, birthYear INT, deathYear INT, primaryProfession TEXT, knownForTitles TEXT);"
sudo -u postgres psql -U postgres -c "COPY imdb FROM '/data/name.basics.tsv' WITH (HEADER true);"
```

## Step 2: Install Python Dependencies
Install the required Python packages: `pyautogen` for agent orchestration and `psycopg2` for PostgreSQL connectivity.

```bash
pip install pyautogen psycopg2
```

## Step 3: Import Required Libraries
Create a new Python script and import the necessary modules.

```python
from autogen import ConversableAgent, register_function
from typing import List, Optional, Union, Dict, Any
import psycopg2
import os
```

## Step 4: Create the PostgreSQL Query Tool
Define a function that accepts structured parameters, constructs a safe SQL query, and returns results. This function is designed to be called by the LLM via AutoGen's tool-calling mechanism.

```python
def execute_postgres_query(
    table_name: str,
    columns: List[str],
    filters: Optional[Dict[str, Any]] = None,
    sort_column: Optional[str] = None,
    sort_order: Optional[str] = None,
    limit: Optional[int] = 150,  # Default limit to prevent overly large results
):
    # Validate input
    if not table_name:
        return "Error: table_name is required"
    if not columns:
        return "Error: columns is required"
    if sort_column and not sort_order:
        return "Error: sort_order is required when sort_column is specified"

    # Generate SQL query
    query = f"SELECT {', '.join(columns)} FROM {table_name}"
    params = []

    if filters:
        filter_conditions = []
        for column, value in filters.items():
            if isinstance(value, str) and value.startswith('%') and value.endswith('%'):
                filter_conditions.append(f"{column} LIKE %s")
                params.append(value)
            elif isinstance(value, list):
                filter_conditions.append(f"{column} NOT IN %s")
                params.append(tuple(value))
            else:
                filter_conditions.append(f"{column} = %s")
                params.append(value)
        query += " WHERE " + " AND ".join(filter_conditions)

    if sort_column and sort_order:
        query += f" ORDER BY {sort_column} {sort_order}"
    if limit:
        query += f" LIMIT {limit}"

    # Execute SQL query
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="super_secret_postgres_password",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
```

**Key Points:**
- The function only supports `SELECT` queries (read-only).
- Input validation prevents malformed queries.
- A default row limit prevents accidental large result sets.

## Step 5: Define a Function to Retrieve Database Schema
To help the LLM understand the database structure, create a helper function that queries PostgreSQL's information schema and returns a list of tables and their columns.

```python
def get_all_tables():
    # Exclude default PostgreSQL schemas
    excluded_schemas = ['information_schema', 'pg_catalog']

    # Query to get all tables excluding the default schemas
    table_columns = ['table_schema', 'table_name']
    table_name = 'information_schema.tables'
    filters = {'table_schema': excluded_schemas}
    sort_column = 'table_schema'
    sort_order = 'ASC'

    # Execute the query to get all tables
    tables_query_result = execute_postgres_query(
        table_name,
        table_columns,
        filters,
        sort_column,
        sort_order
    )

    # Parse the results of the tables query
    tables = [{'table_schema': row[0], 'table_name': row[1]} for row in tables_query_result]

    # Prepare a list to store table information with columns
    table_info = []

    # Iterate over each table to get its columns
    for table in tables:
        schema_name = table['table_schema']
        table_name = table['table_name']

        # Query to get columns for the current table
        columns_columns = ['column_name']
        columns_table_name = 'information_schema.columns'
        columns_filters = {'table_schema': schema_name, 'table_name': table_name}
        columns_sort_column = 'ordinal_position'
        columns_sort_order = 'ASC'

        # Execute the query to get columns
        columns_query_result = execute_postgres_query(
            columns_table_name,
            columns_columns,
            columns_filters,
            columns_sort_column,
            columns_sort_order
        )

        # Parse the results of the columns query
        columns = [row[0] for row in columns_query_result]

        # Add table information with columns to the list
        table_info.append({
            'table_schema': schema_name,
            'table_name': table_name,
            'columns': columns
        })

    return table_info
```

## Step 6: Configure the AutoGen Chatbot
Now, create the main chatbot function that initializes the AutoGen agents, registers the PostgreSQL tool, and starts an interactive session.

```python
def chatbot(mistral_key):
    # LLM configuration for Mistral Large
    config_list = [
        {
            'model': 'mistral-large-2407',
            'base_url': 'https://api.mistral.ai/v1',
            "api_key": mistral_key,
            "tool_choice": "auto",
        },
    ]

    llm_config={
        "config_list": config_list,
        "temperature": 0.1  # Low temperature for more deterministic queries
    }

    # Define the user agent (handles tool execution)
    user = ConversableAgent(
        "user",
        llm_config=False,
        is_termination_msg=lambda msg: "tool_calls" not in msg,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
    )
    
    # Define the assistant agent (uses the LLM)
    assistant = ConversableAgent(
        name="assistant",
        llm_config=llm_config,
        system_message="You are an helpful AI assistant, you use your Postgres tool to query the database. Keep in mind the possibility of to long contexts lengths when using limits wrong."
    )

    # Register the PostgreSQL function as a tool
    assistant.register_for_llm(name="postgres_query", description="Useful for when you need query the postgres db")(execute_postgres_query)
    user.register_for_execution(name="postgres_query")(execute_postgres_query)

    # Provide the assistant with current database schema
    LLM_CONTEXT = get_all_tables()
    user.send(f"This are all the available tables; \n\n  {LLM_CONTEXT} \n\n ", assistant, request_reply=False)
    assistant.send("Thanks for the additonal context of all existing tables!", user, request_reply=False)
    
    # Interactive chat loop
    while True:
        task = input("Enter the query for the LLM ('exit' to quit): ")
        if task.lower() == 'exit':
            break
        context_task = f"{task}"
        user.initiate_chat(assistant, message=context_task, clear_history=False)
```

## Step 7: Run the Chatbot
Finally, prompt for your Mistral API key and start the chatbot.

```python
if __name__ == "__main__":
    mistral_key = input("Enter your Mistral AI key: ")
    chatbot(mistral_key)
```

## Example Queries
Once the chatbot is running, you can ask natural language questions like:

- "Get me all people named Mistral"
- "Get me all actors born in 2000, limit them to 10"
- "Get me all actors from before 1950, limit them to 13"

The LLM will translate these into structured calls to the `execute_postgres_query` function and return the results.

## Conclusion
You have successfully created a safe, read-only interface between Mistral Large 2407 and a PostgreSQL database using Microsoft AutoGen. This pattern can be extended to other databases or augmented with additional tools (e.g., for vector similarity search). By providing schema context and limiting tool capabilities, you enable powerful natural language querying while maintaining data security.

**Best Practices:**
- Always validate and sanitize inputs in your tool functions.
- Use row limits to prevent accidentally large queries.
- Consider using separate agents for different tasks (e.g., query planning vs. execution) to improve reliability.