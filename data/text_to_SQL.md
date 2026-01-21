# Building a Scalable Text-to-SQL Agent with Mistral AI

This guide will walk you through building an intelligent agent that can answer natural language questions about a multi-table SQL database. You'll learn to use Mistral's function-calling capabilities to create a scalable Text-to-SQL system that dynamically retrieves only the necessary schema information, avoiding the performance pitfalls of injecting entire database schemas into prompts.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- A Mistral AI API key
- Basic familiarity with SQL and Python

## Setup

First, install the required packages:

```bash
pip install mistralai langchain deepeval
```

Now, import the necessary modules:

```python
from mistralai import Mistral
from getpass import getpass
from langchain_community.utilities import SQLDatabase
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
import json
```

## Step 1: Set Up the Chinook Database

The Chinook database is a sample database representing a digital media store. It contains tables for artists, albums, tracks, invoices, and customers—perfect for testing Text-to-SQL systems.

### Install SQLite

If you don't have SQLite installed, install it first:

```bash
sudo apt install sqlite3
```

### Download and Create the Database

Create the Chinook database by running:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
```

This command downloads the SQL script and pipes it directly into SQLite to create the database.

## Step 2: Initialize the Mistral Client

Set up your Mistral client with your API key:

```python
api_key = getpass("Enter your Mistral AI API Key: ")
client = Mistral(api_key=api_key)
uri = "sqlite:///Chinook.db"
```

## Step 3: Create Database Interaction Functions

You'll need two core functions to interact with the database: one to execute queries and another to retrieve table schemas.

### Function 1: Execute SQL Queries

This function runs SQL code against the Chinook database and returns the results:

```python
def run_sql_query(sql_code):
    """
    Executes the given SQL query against the database and returns the results.

    Args:
        sql_code (str): The SQL query to be executed.

    Returns:
        result: The results of the SQL query.
    """
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    return db.run(sql_code)

# Test the function
run_sql_query("SELECT * FROM Artist LIMIT 10;")
```

### Function 2: Retrieve Table Schemas

Instead of loading all table schemas at once, this function dynamically retrieves the schema for a specific table when needed. This approach is more efficient than injecting all schemas into the prompt.

```python
def get_sql_schema_of_table(table):
    """
    Returns the schema of a table.

    Args:
        table (str): Name of the table to be described

    Returns:
        result: Column names and types of the table
    """
    # Return the CREATE TABLE statement for the requested table
    if table == "Album":
        return """ The table Album was created with the following code :

CREATE TABLE [Album]
(
    [AlbumId] INTEGER  NOT NULL,
    [Title] NVARCHAR(160)  NOT NULL,
    [ArtistId] INTEGER  NOT NULL,
    CONSTRAINT [PK_Album] PRIMARY KEY  ([AlbumId]),
    FOREIGN KEY ([ArtistId]) REFERENCES [Artist] ([ArtistId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """
    
    # Similar if-else blocks for other tables...
    # (Artist, Customer, Employee, Genre, Invoice, InvoiceLine, 
    #  MediaType, Playlist, PlaylistTrack, Track)
    
    return f"The table {table} does not exist in the Chinook database"
```

**Note:** The complete function includes similar blocks for all 11 tables in the Chinook database. This selective retrieval prevents token waste and improves performance.

## Step 4: Build the Text-to-SQL Agent

Now, create the main agent function that coordinates between the user's question, the LLM, and the database tools.

### Define the Agent Function

```python
def get_response(question, verbose=True):
    """
    Answer question about the Chinook database.

    Args:
        question (str): The question asked by the user.
        verbose (bool): If True, prints intermediate steps and results.

    Returns:
        str: The response to the user's question.
    """

    # Define the tools available for the AI assistant
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_sql_schema_of_table",
                "description": "Get the schema of a table in the Chinook database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "enum": ["Album", "Artist", "Customer", "Employee", "Genre", 
                                     "Invoice", "InvoiceLine", "MediaType", "Playlist", 
                                     "PlaylistTrack", "Track"],
                            "description": "The question asked by the user",
                        },
                    },
                    "required": ["table"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_sql_query",
                "description": "Run an SQL query on the Chinook database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_code": {
                            "type": "string",
                            "description": "SQL code to be run",
                        },
                    },
                    "required": ["sql_code"],
                },
            },
        }
    ]

    # System prompt for the AI assistant
    system_prompt = """
    You are an AI assistant.
    Your job is to reply to questions related to the Chinook database.
    The Chinook data model represents a digital media store, including tables for artists, albums, media tracks, invoices, and customers.

    To answer user questions, you have two tools at your disposal.

    Firstly, a function called "get_sql_schema_of_table" which has a single parameter named "table" whose value is an element
    of the following list: ["Album", "Artist", "Customer", "Employee", "Genre", "Invoice", "InvoiceLine", "MediaType", "Playlist", "PlaylistTrack", "Track"].

    Secondly, a function called "run_sql_query" which has a single parameter named "sql_code".
    It will run SQL code on the Chinook database. The SQL dialect is SQLite.

    For a given question, your job is to:
    1. Get the schemas of the tables that might help you answer the question using the "get_sql_schema_of_table" function.
    2. Run a SQLite query on the relevant tables using the "run_sql_query" function.
    3. Answer the user based on the result of the SQL query.

    You will always remain factual, you will not hallucinate, and you will say that you don't know if you don't know.
    You will politely ask the user to shoot another question if the question is not related to the Chinook database.
    """

    # Initialize chat history with system prompt and user question
    chat_history = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    if verbose:
        print(f"User: {question}\n")

    used_run_sql = False
    used_get_sql_schema_of_table = False

    # Function to determine tool choice based on usage
    def tool_choice(used_run_sql, used_get_sql_schema_of_table):
        # If the question is out of topic the agent is not expected to run a tool call
        if not used_get_sql_schema_of_table:
            return "auto"
        # The agent is expected to run "used_run_sql" after getting the specifications of the tables of interest
        if used_get_sql_schema_of_table and not used_run_sql:
            return "any"
        # The agent is not expected to run a tool call after querying the SQL table
        if used_run_sql and used_get_sql_schema_of_table:
            return "none"
        return "auto"

    iteration = 0
    max_iteration = 10

    # Main loop to process the question
    while iteration < max_iteration:
        inference = client.chat.complete(
            model="mistral-large-latest",
            temperature=0.3,
            messages=chat_history,
            tools=tools,
            tool_choice=tool_choice(used_run_sql, used_get_sql_schema_of_table)
        )

        chat_history.append(inference.choices[0].message)

        tool_calls = inference.choices[0].message.tool_calls

        if not tool_calls:
            if verbose:
                print(f"Assistant: {inference.choices[0].message.content}\n")
            return inference.choices[0].message.content

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)

            if function_name == "get_sql_schema_of_table":
                function_result = get_sql_schema_of_table(function_params['table'])
                if verbose:
                    print(f"Tool: Getting SQL schema of table {function_params['table']}\n")
                used_get_sql_schema_of_table = True

            if function_name == "run_sql_query":
                function_result = run_sql_query(function_params['sql_code'])
                if verbose:
                    print(f"Tool: Running code {function_params['sql_code']}\n")
                used_run_sql = True

            chat_history.append({"role": "tool", "name": function_name, "content": function_result, "tool_call_id": tool_call.id})

        iteration += 1
    return
```

### How the Agent Works

The agent follows a logical three-step process:

1. **Schema Retrieval**: When presented with a question, the agent first identifies which tables might be relevant and retrieves only their schemas using `get_sql_schema_of_table`.

2. **Query Generation and Execution**: With the necessary schema information, the agent constructs an appropriate SQL query and executes it using `run_sql_query`.

3. **Response Formulation**: Finally, the agent interprets the query results and formulates a natural language response.

The `tool_choice` function intelligently guides the LLM through this sequence, ensuring it doesn't attempt to run queries before understanding the database structure.

## Step 5: Test the Agent

Let's test the agent with questions of increasing complexity:

```python
# Test with an out-of-topic question
response = get_response('What is the oldest player in the NBA?')
```

The agent should recognize this is unrelated to the Chinook database and respond accordingly.

```python
# Simple query about genres
response = get_response('What are the genre in the store?')
```

```python
# Query requiring table joins
response = get_response('What are the albums of the rock band U2?')
```

```python
# More complex analytical query
response = get_response('What is the shortest song that the rock band U2 ever composed?')
```

```python
# Complex query with aggregation
response = get_response('Which track from U2 is the most sold?')
```

```python
# Multi-table analytical query
response = get_response('Which consumer bought the biggest amount of U2 songs?')
```

```python
# Pattern matching query
response = get_response('List all artist that have a color in their name')
```

```python
# Business intelligence query
response = get_response('Who are our top Customers according to Invoices?')
```

## Step 6: Evaluate Performance with DeepEval

For systematic evaluation, you can use the DeepEval framework with Mistral as the judge model.

### Create a Custom Mistral Model Wrapper

```python
class CustomMistralLarge(DeepEvalBaseLLM):
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        self.model_name = "mistral-large-latest"

    def get_model_name(self):
        return "Mistral-large-latest"

    def load_model(self):
        # Since we are using the Mistral API, we don't need to load a model object.
        return self.client

    def generate(self, prompt: str) -> str:
        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        return chat_response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # Reusing the synchronous generate method for simplicity.
        return self.generate(prompt)
```

### Create an Evaluation Dataset

You can build a test set based on questions from resources like [this Chinook database article](https://medium.com/@raufrukayat/chinook-database-querying-a-digital-music-store-database-8c98cf0f8611):

```python
# Specify evaluation questions
questions = [
    "Which Employee has the Highest Total Number of Customers?",
    "Who are our top Customers according to Invoices?",
    "How many tracks are there in the database?",
    # Add more questions as needed
]
```

With DeepEval, you can then create test cases, run evaluations, and measure metrics like accuracy, relevance, and coherence of the agent's responses.

## Key Takeaways

1. **Selective Schema Retrieval**: By dynamically fetching only relevant table schemas, you avoid token waste and improve performance compared to injecting entire database schemas into prompts.

2. **Structured Tool Usage**: The agent's three-step process (schema retrieval → query execution → response formulation) ensures reliable and accurate responses.

3. **Intelligent Tool Guidance**: The `tool_choice` function helps guide the LLM through the correct sequence of operations.

4. **Scalable Architecture**: This approach scales well to databases with many tables, as you only pay the token cost for the schemas actually needed to answer each question.

This Text-to-SQL agent demonstrates how to leverage Mistral's function-calling capabilities to build efficient, scalable database querying systems that can handle complex natural language questions across multi-table databases.