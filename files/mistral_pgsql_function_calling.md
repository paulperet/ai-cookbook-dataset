# Microsoft AutoGen & Mistral Large 2407 - Retrieve information from a Postgresql Database

This tutorial is written as example how to use Microsoft AutoGen in Combination with Mistral Large V2 to query a Postgres Database.
It gives a short and simple overview how to do this, it is also possible to write extra tools and add them to query other DB's or maybe VectorDB's with a perfect prompt ;)! 

Besides query's and prompting a good context is also really important, thats why I added a extra function to always provide the context of all avaible tables.

**PLEASE NOTE:** Try to prevent using to many tools and contexts together, but use different 'chat models' instead of a single big model to do everything.

## Preparation steps:

In the first few steps we will install Postgresql, after that we will import a database dumb.
I did go for the following database dumb; name.basics.tsv, found here; https://datasets.imdbws.com/

- https://wiki.postgresql.org/wiki/Sample_Databases

```python
!sudo apt update
!sudo apt install dirmngr ca-certificates software-properties-common gnupg gnupg2 apt-transport-https curl -y
!curl -fSsL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor | sudo tee /usr/share/keyrings/postgresql.gpg > /dev/null
!echo 'deb [arch=amd64,arm64,ppc64el signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt/ jammy-pgdg main' | sudo tee /etc/apt/sources.list.d/pgdg.list
!sudo apt update
!sudo apt install postgresql-client-16 postgresql-16 -y
!sudo service postgresql start
```

```python
!sudo -u postgres psql -U postgres -c "ALTER ROLE postgres WITH PASSWORD 'super_secret_postgres_password';"
```

```python
# Download the dataset from IMDB
!cd /
!sudo mkdir data
!sudo wget https://datasets.imdbws.com/name.basics.tsv.gz
!sudo gunzip name.basics.tsv.gz

# Create a super user
!sudo -u postgres psql -U postgres -c "CREATE ROLE root WITH SUPERUSER;"
!sudo -u postgres psql -U postgres -c "ALTER ROLE root WITH LOGIN;"
!sudo -u postgres psql -U postgres -c "CREATE ROLE postgres WITH PASSWORD 'super_secret_postgres_password';"

# Import the dataset
!sudo -u postgres psql -U postgres -c "CREATE TABLE imdb ( nconst TEXT, primaryName TEXT, birthYear INT, deathYear INT, primaryProfession TEXT, knownForTitles TEXT);"

#It is possible you have to change the directory which containts this file
!sudo -u postgres psql -U postgres -c "COPY imdb FROM '/content/name.basics.tsv' WITH (HEADER true);"
```

```python
!pip install pyautogen psycopg2
```

# Import all required packages

Lets importat all required packages, in this case we need autogen, the postgresql package and some other libraries.

```python
from autogen import ConversableAgent, register_function
from typing import List, Optional, Union, Dict, Any
import psycopg2
import os
```

# The Postgress Function

The Postgress function takes a valid json input, based on this input the query is executed.
After that the function returns the output to the LLM, which will respond on that with a message to the user.

Because we pre-defined how to use the tool, it is not possible to delete , update or create any records, read access only!

```python
def execute_postgres_query(
    table_name: str,
    columns: List[str],
    filters: Optional[Dict[str, Any]] = None,
    sort_column: Optional[str] = None,
    sort_order: Optional[str] = None,
    limit: Optional[int] = 150,  # Default limit of 150 rows, you can edit this yourself if needed, the AI will also be able to change this.
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
    conn = psycopg2.connect(database="postgres", user="postgres", password="super_secret_postgres_password", host="localhost", port="5432")
    cur = conn.cursor()
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
```

# Define API keys

Fill in your Mistral API key to access Mistral-large-2407! :D

```python
mistral_key = input("Enter your Mistral AI key: ")
```

# Get All tables

We want to prompt the LLM with the context of all tables, this has to be up to date, so we create a sepperate function which queries the postgres tool but with pre defined input to query all tables.

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

# Execute user queries

Now everything is set to use the chat and query Postgresql Database using Mistral-Large-2407

```python
def chatbot(mistral_key):
    config_list = [
        {
            'model': 'mistral-large-2407', # If the responses are very slow, change this model to open-mixtral-8x22b
            'base_url': 'https://api.mistral.ai/v1',
            "api_key": mistral_key,
            "tool_choice": "auto",
        },
    ]

    llm_config={
        "config_list": config_list,
        "temperature": 0.1
    }

    user = ConversableAgent(
        "user",
        llm_config=False,
        is_termination_msg=lambda msg: "tool_calls" not in msg,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
    )
    
    assistant = ConversableAgent(
    name="assistant",
        llm_config=llm_config,
        system_message="You are an helpful AI assistant, you use your Postgres tool to query the database. Keep in mind the possibility of to long contexts lengths when using limits wrong."
    )

    assistant.register_for_llm(name="postgres_query", description="Useful for when you need query the postgres db")(execute_postgres_query)
    user.register_for_execution(name="postgres_query")(execute_postgres_query)

    LLM_CONTEXT = get_all_tables()

    user.send(f"This are all the available tables; \n\n  {LLM_CONTEXT} \n\n ", assistant, request_reply=False)
    assistant.send("Thanks for the additonal context of all existing tables!", user, request_reply=False)
    
    while True:
        task = input("Enter the query for the LLM ('exit' to quit): ")
        if task.lower() == 'exit':
            break
        context_task = f"{task}"
        user.initiate_chat(assistant, message=context_task, clear_history=False)

chatbot(mistral_key)
```

# Example questions & Conclusion

Some good example questions to ask this model are;
- Get me all people named Mistral
- Get me all actors born in 2000, limit them to 10
- Get me all actors from before 1950, limit them to 13

So like you can see Mistral Large V2 or any equivelant model it is relatively easy to create a function to query a Postgres DB without giving it full access to delete records. 

This can give people a safer way to access different databases, without having to worry to make mistakes.