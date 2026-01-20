# How to call functions with chat models

This notebook covers how to use the Chat Completions API in combination with external functions to extend the capabilities of GPT models.

`tools` is an optional parameter in the Chat Completion API which can be used to provide function specifications. The purpose of this is to enable models to generate function arguments which adhere to the provided specifications. Note that the API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.

Within the `tools` parameter, if the `functions` parameter is provided then by default the model will decide when it is appropriate to use one of the functions. The API can be forced to use a specific function by setting the `tool_choice` parameter to `{"type": "function", "function": {"name": "my_function"}}`. The API can also be forced to not use any function by setting the `tool_choice` parameter to `"none"`. If a function is used, the output will contain `"finish_reason": "tool_calls"` in the response, as well as a `tool_calls` object that has the name of the function and the generated function arguments.

### Overview

This notebook contains the following 2 sections:

- **How to generate function arguments:** Specify a set of functions and use the API to generate function arguments.
- **How to call functions with model generated arguments:** Close the loop by actually executing functions with model generated arguments.

## How to generate function arguments

```python
!pip install scipy --quiet
!pip install tenacity --quiet
!pip install tiktoken --quiet
!pip install termcolor --quiet
!pip install openai --quiet
```

```python
import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored  

GPT_MODEL = "gpt-5"
client = OpenAI()
```

### Utilities

First let's define a few utilities for making calls to the Chat Completions API and for maintaining and keeping track of the conversation state.

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

### Basic concepts

Let's create some function specifications to interface with a hypothetical weather API. We'll pass these function specification to the Chat Completions API in order to generate function arguments that adhere to the specification.

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

If we prompt the model about the current weather, it will respond with some clarifying questions.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(
    messages, tools=tools
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)

```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: What's the weather like today

    assistant: Sure—what city and state (or country) should I check? Also, do you prefer Celsius or Fahrenheit?


Once we provide the missing information, it will generate the appropriate function arguments for us.

```python
messages.append({"role": "user", "content": "I'm in Glasgow, Scotland."})
chat_response = chat_completion_request(
    messages, tools=tools
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)

```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: What's the weather like today

    assistant: Sure—what city and state (or country) should I check? Also, do you prefer Celsius or Fahrenheit?

    user: I'm in Glasgow, Scotland.

    assistant: [{'id': 'call_k2QgGc9GT9WjxD76GvR0Ot8q', 'function': {'arguments': '{"location": "Glasgow, Scotland", "format": "celsius"}', 'name': 'get_current_weather'}, 'type': 'function'}, {'id': 'call_RtnXV5t49lqbWwhvGoEPZ7KY', 'function': {'arguments': '{"location": "Glasgow, Scotland", "format": "celsius", "num_days": 1}', 'name': 'get_n_day_weather_forecast'}, 'type': 'function'}]


By prompting it differently, we can get it to target the other function we've told it about.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in Glasgow, Scotland over the next x days"})
chat_response = chat_completion_request(
    messages, tools=tools
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)

```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: what is the weather going to be like in Glasgow, Scotland over the next x days

    assistant: How many days would you like the forecast for in Glasgow, Scotland? For example: 3, 5, 7, 10, or 14.


Once again, the model is asking us for clarification because it doesn't have enough information yet. In this case it already knows the location for the forecast, but it needs to know how many days are required in the forecast.

```python
messages.append({"role": "user", "content": "5 days"})
chat_response = chat_completion_request(
    messages, tools=tools
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)

```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: what is the weather going to be like in Glasgow, Scotland over the next x days

    assistant: How many days would you like the forecast for in Glasgow, Scotland? For example: 3, 5, 7, 10, or 14.

    user: 5 days

    assistant: [{'id': 'call_lNzOVLrNSaSVjL3O3bN110af', 'function': {'arguments': '{"location":"Glasgow, Scotland","format":"celsius","num_days":5}', 'name': 'get_n_day_weather_forecast'}, 'type': 'function'}]


#### Forcing the use of specific functions or no function

We can force the model to use a specific function, for example get_n_day_weather_forecast by using the function_call argument. By doing so, we force the model to make assumptions about how to use it.

```python
# in this cell we force the model to use get_n_day_weather_forecast
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools, tool_choice={"type": "function", "function": {"name": "get_n_day_weather_forecast"}}
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: Give me a weather report for Toronto, Canada.

    assistant: [{'id': 'call_3hoMjl55OQ7LxfwhFyjxwv1T', 'function': {'arguments': '{"location":"Toronto, Canada","format":"celsius","num_days":5}', 'name': 'get_n_day_weather_forecast'}, 'type': 'function'}]

```python
# if we don't force the model to use get_n_day_weather_forecast it may not
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: Give me a weather report for Toronto, Canada.

    assistant: [{'id': 'call_wv5mdjEQJnBPuSci3xw09Tom', 'function': {'arguments': '{"location":"Toronto, ON","format":"celsius"}', 'name': 'get_current_weather'}, 'type': 'function'}]


We can also force the model to not use a function at all. By doing so we prevent it from producing a proper function call.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me the current weather (use Celcius) for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools, tool_choice="none"
)
messages.append(chat_response.choices[0].message.to_dict())
pretty_print_conversation(messages)
```

    system: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.

    user: Give me the current weather (use Celcius) for Toronto, Canada.

    assistant: I don’t have live access to pull the current conditions right now. To get Toronto’s current weather in Celsius, check any of these quickly:
    - Environment Canada: weather.gc.ca (search “Toronto”)
    - The Weather Network: theweathernetwork.com/ca/weather/ontario/toronto
    - Google: search “Toronto weather” (shows °C by default in Canada)
    - AccuWeather or Weather.com (set units to °C)
    
    If you paste the current readings here (temperature, feels-like, wind, precipitation), I can interpret them and advise on what to wear or plan for.

### Parallel Function Calling

Newer models such as gpt-5, gpt-4.1 or gpt-4o can call multiple functions in one turn.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days"})
chat_response = chat_completion_request(
    messages, tools=tools, model="gpt-4o"
)

assistant_message = chat_response.choices[0].message.tool_calls
assistant_message
```




    [ChatCompletionMessageFunctionToolCall(id='call_KlZ3Fqt3SviC6o66dVMYSa2Q', function=Function(arguments='{"location": "San Francisco, CA", "format": "fahrenheit", "num_days": 4}', name='get_n_day_weather_forecast'), type='function'),
     ChatCompletionMessageFunctionToolCall(id='call_YAnH0VRB3oqjqivcGj3Cd8YA', function=Function(arguments='{"location": "Glasgow, UK", "format": "celsius", "num_days": 4}', name='get_n_day_weather_forecast'), type='function')]



## How to call functions with model generated arguments

In our next example, we'll demonstrate how to execute functions whose inputs are model-generated, and use this to implement an agent that can answer questions for us about a database. For simplicity we'll use the [Chinook sample database](https://www.sqlitetutorial.net/sqlite-sample-database/).

*Note:* SQL generation can be high-risk in a production environment since models are not perfectly reliable at generating correct SQL.

### Specifying a function to execute SQL queries

First let's define some helpful utility functions to extract data from a SQLite database.

```python
import sqlite3

conn = sqlite3.connect("data/Chinook.db")
print("Opened database successfully")
```

    Opened database successfully

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

Now we can use these utility functions to extract a representation of the database schema.

```python
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)
```

As before, we'll define a function specification for the function we'd like the API to generate arguments for. Notice that we are inserting the database schema into the function specification. This will be important for the model to know about.

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

### Executing SQL queries

Now let's implement the function that will actually excute queries against the database.

```python
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results
```

##### Steps to invoke a function call using Chat