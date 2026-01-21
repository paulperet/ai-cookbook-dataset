# Guide: Calling Functions with Chat Models

This guide demonstrates how to use the OpenAI Chat Completions API with external function specifications. You'll learn to generate structured function arguments from natural language and execute those functions to build a simple agent.

## Prerequisites

Install the required packages and set up your environment.

```bash
pip install openai tenacity tiktoken termcolor scipy
```

```python
import json
import sqlite3
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-5"
client = OpenAI()
```

## Step 1: Define Core Utilities

First, create helper functions for API calls and conversation logging.

### 1.1 Retry-Enabled API Call

This function handles Chat Completion requests with automatic retries.

```python
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
```

### 1.2 Conversation Logger

This utility prints a color-coded conversation history.

```python
def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("tool_calls"):
            print(colored(f"assistant: {message['tool_calls']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("tool_calls"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))
```

## Step 2: Generate Function Arguments

Define function specifications and let the model generate structured arguments.

### 2.1 Define Weather Functions

Create two hypothetical weather API functions.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this unit from the forecast location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this unit from the forecast location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]
```

### 2.2 Test Basic Function Argument Generation

Start a conversation and see how the model asks for clarification.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(messages, tools=tools)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

The model will respond with clarifying questions. Provide the missing information.

```python
messages.append({"role": "user", "content": "I'm in Glasgow, Scotland."})
chat_response = chat_completion_request(messages, tools=tools)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

Now the model generates function arguments for both weather functions.

### 2.3 Target Specific Functions

You can prompt the model to use a specific function.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in Glasgow, Scotland over the next x days"})
chat_response = chat_completion_request(messages, tools=tools)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

The model asks for the number of days. Provide it.

```python
messages.append({"role": "user", "content": "5 days"})
chat_response = chat_completion_request(messages, tools=tools)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

### 2.4 Control Function Usage with `tool_choice`

Force the model to use a specific function.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools, tool_choice={"type": "function", "function": {"name": "get_n_day_weather_forecast"}}
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

Force the model to not use any function.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me the current weather (use Celcius) for Toronto, Canada."})
chat_response = chat_completion_request(messages, tools=tools, tool_choice="none")
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

### 2.5 Parallel Function Calling

Newer models (gpt-4o, gpt-4.1, gpt-5) can call multiple functions in a single turn.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days"})
chat_response = chat_completion_request(messages, tools=tools, model="gpt-4o")
assistant_message = chat_response.choices[0].message.tool_calls
print(assistant_message)
```

## Step 3: Execute Functions with Model-Generated Arguments

Now, implement an agent that answers questions by querying a SQL database.

### 3.1 Set Up the Database

Use the Chinook sample database.

```python
conn = sqlite3.connect("data/Chinook.db")
print("Opened database successfully")
```

### 3.2 Create Database Utility Functions

These functions extract schema information.

```python
def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names

def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names

def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts
```

Extract the database schema.

```python
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)
```

### 3.3 Define the SQL Query Function

Create a function specification that includes the database schema.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Use this function to answer user questions about music. Input should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    }
]
```

### 3.4 Implement the Query Execution Function

This function runs the generated SQL.

```python
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results
```

## Step 4: Build the Agent Loop

Combine everything to create an agent that processes user questions, generates SQL, executes it, and returns answers.

### 4.1 Agent Execution Steps

1. **Initialize** the conversation with a system message.
2. **Receive** a user question.
3. **Generate** a function call using the Chat Completions API.
4. **Extract** the SQL query from the model's response.
5. **Execute** the query using `ask_database`.
6. **Return** the results to the conversation.
7. **Continue** until the question is answered.

Here's a conceptual outline:

```python
def run_agent(user_question, conn, tools):
    messages = [
        {"role": "system", "content": "Answer user questions by generating SQL queries against the database."},
        {"role": "user", "content": user_question}
    ]
    
    # Step 1: Generate SQL query
    response = chat_completion_request(messages, tools=tools)
    assistant_message = response.choices[0].message
    
    if assistant_message.tool_calls:
        # Step 2: Extract and execute query
        tool_call = assistant_message.tool_calls[0]
        query = json.loads(tool_call.function.arguments)["query"]
        results = ask_database(conn, query)
        
        # Step 3: Add results to conversation
        messages.append(assistant_message.to_dict())
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": results
        })
        
        # Step 4: Get final answer
        final_response = chat_completion_request(messages, tools=tools)
        return final_response.choices[0].message.content
    
    return assistant_message.content
```

### 4.2 Example Usage

```python
question = "Which artists have the most albums?"
answer = run_agent(question, conn, tools)
print(answer)
```

## Summary

You've learned how to:
- Define function specifications for the Chat Completions API.
- Generate structured function arguments from natural language.
- Control function usage with the `tool_choice` parameter.
- Execute parallel function calls with newer models.
- Build an agent that queries a SQL database using model-generated SQL.

Remember: Always validate and sanitize model-generated code (like SQL) before execution in production environments.