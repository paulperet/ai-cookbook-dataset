# Text to SQL on multi-tables database

In this cookbook we will show you how to :

- Use the function calling capabilities of Mistral models
- Build a text2SQL architecture that scales more efficiently than a naive approach where all schemas are integrally injected in the system prompt  
- Evaluate your system with Mistral models and leveraging the DeepEval framework

# Imports

```python
!pip install mistralai langchain deepeval
```

[First Entry, ..., Last Entry]

```python
from mistralai import Mistral
from getpass import getpass

# To interract with the SQL database
from langchain_community.utilities import SQLDatabase

# To evaluate text2SQL performances
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval

import json
```

# Load the Chinook database

"Chinook is a sample database available for SQL Server, Oracle, MySQL, etc. It can be created by running a single SQL script. Chinook database is an alternative to the Northwind database, being ideal for demos and testing ORM tools targeting single and multiple database servers."

To run this notebook you will need to download the Chinook datase. You will find more information about this database by clicking on this [github link](https://github.com/lerocha/chinook-database).

To create the `Chinook.db` in the same directory as this notebook you have several options :

- You can download and build the database via the command line :

```
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
```

- Another strategy consists in running the following script `https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql`.

Firstly save the script to a folder/directory on your computer. Then create a database called Chinook with `sqlite3 Chinook.db`. Ultimately run the script with the command `.read Chinook_Sqlite.sql`

```python
!sudo apt install sqlite3
```

[First Entry, ..., Last Entry]

```python
!curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

```python
!sqlite3 Chinook.db
```

[First Entry, ..., Last Entry]

 # Set up clients

```python
api_key= getpass("Type your API Key")
client = Mistral(api_key=api_key)
uri = "sqlite:///Chinook.db"
```

# Interract with the Chinook database

We are defining two functions :
- run_sql_query that runs sql code on Chinook
- get_sql_schema_of_table that returns the schema of a table specified as input

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

run_sql_query("SELECT * FROM Artist LIMIT 10;")
```

```python
def get_sql_schema_of_table(table):
    """
    Returns the schema of a table.

    Args:
        table (str): Name of the table to be described

    Returns:
        result: Column names and types of the table
    """
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

    if table == "Artist":
        return """ The table Artist was created with the following code :

CREATE TABLE [Artist]
(
    [ArtistId] INTEGER  NOT NULL,
    [Name] NVARCHAR(120),
    CONSTRAINT [PK_Artist] PRIMARY KEY  ([ArtistId])
);
        """

    if table == "Customer":
        return """ The table Customer was created with the following code :

CREATE TABLE [Customer]
(
    [CustomerId] INTEGER  NOT NULL,
    [FirstName] NVARCHAR(40)  NOT NULL,
    [LastName] NVARCHAR(20)  NOT NULL,
    [Company] NVARCHAR(80),
    [Address] NVARCHAR(70),
    [City] NVARCHAR(40),
    [State] NVARCHAR(40),
    [Country] NVARCHAR(40),
    [PostalCode] NVARCHAR(10),
    [Phone] NVARCHAR(24),
    [Fax] NVARCHAR(24),
    [Email] NVARCHAR(60)  NOT NULL,
    [SupportRepId] INTEGER,
    CONSTRAINT [PK_Customer] PRIMARY KEY  ([CustomerId]),
    FOREIGN KEY ([SupportRepId]) REFERENCES [Employee] ([EmployeeId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """

    if table == "Employee":
        return """ The table Employee was created with the following code :

CREATE TABLE [Employee]
(
    [EmployeeId] INTEGER  NOT NULL,
    [LastName] NVARCHAR(20)  NOT NULL,
    [FirstName] NVARCHAR(20)  NOT NULL,
    [Title] NVARCHAR(30),
    [ReportsTo] INTEGER,
    [BirthDate] DATETIME,
    [HireDate] DATETIME,
    [Address] NVARCHAR(70),
    [City] NVARCHAR(40),
    [State] NVARCHAR(40),
    [Country] NVARCHAR(40),
    [PostalCode] NVARCHAR(10),
    [Phone] NVARCHAR(24),
    [Fax] NVARCHAR(24),
    [Email] NVARCHAR(60),
    CONSTRAINT [PK_Employee] PRIMARY KEY  ([EmployeeId]),
    FOREIGN KEY ([ReportsTo]) REFERENCES [Employee] ([EmployeeId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """

    if table == "Genre":
        return """ The table Genre was created with the following code :

 CREATE TABLE [Genre]
(
    [GenreId] INTEGER  NOT NULL,
    [Name] NVARCHAR(120),
    CONSTRAINT [PK_Genre] PRIMARY KEY  ([GenreId])
);
        """

    if table == "Invoice":
        return """ The table Invoice was created with the following code :

CREATE TABLE [Invoice]
(
    [InvoiceId] INTEGER  NOT NULL,
    [CustomerId] INTEGER  NOT NULL,
    [InvoiceDate] DATETIME  NOT NULL,
    [BillingAddress] NVARCHAR(70),
    [BillingCity] NVARCHAR(40),
    [BillingState] NVARCHAR(40),
    [BillingCountry] NVARCHAR(40),
    [BillingPostalCode] NVARCHAR(10),
    [Total] NUMERIC(10,2)  NOT NULL,
    CONSTRAINT [PK_Invoice] PRIMARY KEY  ([InvoiceId]),
    FOREIGN KEY ([CustomerId]) REFERENCES [Customer] ([CustomerId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """

    if table == "InvoiceLine":
        return """ The table InvoiceLine was created with the following code :

CREATE TABLE [InvoiceLine]
(
    [InvoiceLineId] INTEGER  NOT NULL,
    [InvoiceId] INTEGER  NOT NULL,
    [TrackId] INTEGER  NOT NULL,
    [UnitPrice] NUMERIC(10,2)  NOT NULL,
    [Quantity] INTEGER  NOT NULL,
    CONSTRAINT [PK_InvoiceLine] PRIMARY KEY  ([InvoiceLineId]),
    FOREIGN KEY ([InvoiceId]) REFERENCES [Invoice] ([InvoiceId])
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([TrackId]) REFERENCES [Track] ([TrackId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """


    if table == "MediaType":
        return """ The table MediaType was created with the following code :

CREATE TABLE [MediaType]
(
    [MediaTypeId] INTEGER  NOT NULL,
    [Name] NVARCHAR(120),
    CONSTRAINT [PK_MediaType] PRIMARY KEY  ([MediaTypeId])
);
        """

    if table == "Playlist":
        return """ The table Playlist was created with the following code :

CREATE TABLE [Playlist]
(
    [PlaylistId] INTEGER  NOT NULL,
    [Name] NVARCHAR(120),
    CONSTRAINT [PK_Playlist] PRIMARY KEY  ([PlaylistId])
);
        """

    if table == "PlaylistTrack":
        return """ The table PlaylistTrack was created with the following code :

CREATE TABLE [PlaylistTrack]
(
    [PlaylistId] INTEGER  NOT NULL,
    [TrackId] INTEGER  NOT NULL,
    CONSTRAINT [PK_PlaylistTrack] PRIMARY KEY  ([PlaylistId], [TrackId]),
    FOREIGN KEY ([PlaylistId]) REFERENCES [Playlist] ([PlaylistId])
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([TrackId]) REFERENCES [Track] ([TrackId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """

    if table == "Track":
        return """ The table Track was created with the following code :

CREATE TABLE [Track]
(
    [TrackId] INTEGER  NOT NULL,
    [Name] NVARCHAR(200)  NOT NULL,
    [AlbumId] INTEGER,
    [MediaTypeId] INTEGER  NOT NULL,
    [GenreId] INTEGER,
    [Composer] NVARCHAR(220),
    [Milliseconds] INTEGER  NOT NULL,
    [Bytes] INTEGER,
    [UnitPrice] NUMERIC(10,2)  NOT NULL,
    CONSTRAINT [PK_Track] PRIMARY KEY  ([TrackId]),
    FOREIGN KEY ([AlbumId]) REFERENCES [Album] ([AlbumId])
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([GenreId]) REFERENCES [Genre] ([GenreId])
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY ([MediaTypeId]) REFERENCES [MediaType] ([MediaTypeId])
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
        """

    return f"The table {table} does not exist in the Chinook database"
```

# Build agent

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
                            "enum": ["Album", "Artist", "Customer", "Employee", "Genre", "Invoice", "InvoiceLine", "MediaType", "Playlist", "PlaylistTrack", "Track"],
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

# Test the agent

Let's test the agent and ask a few random questions of increasing complexity

```python
# Lets start by checking how the model reacts with out of topic questions!
response = get_response('What is the oldest player in the NBA?')
```

```python
response = get_response('What are the genre in the store?')
```

```python
response = get_response('What are the albums of the rock band U2?')
```

```python
response = get_response('What is the shortest song that the rock band U2 ever composed?')
```

```python
response = get_response('Which track from U2 is the most sold?')
```

```python
response = get_response('Which consumer bought the biggest amound of U2 songs?')
```

```python
response = get_response('List all artist that have a color in their name')
```

```python
response = get_response('Who are our top Customers according to Invoices?')
```

# Evaluating

Let's try to evaluate the agent in a more formal way.

We will build a test set based on the questions from this Medium article [Chinook question/answers](https://medium.com/@raufrukayat/chinook-database-querying-a-digital-music-store-database-8c98cf0f8611)

We will evaluate answers via LLM as a judge through the framework [DeepEval](https://docs.confident-ai.com/) from which the image here below is taken.

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

```python
# Specify questions
questions = [
    "Which Employee has the Highest Total Number of Customers?",
    "Who are our top Customers according to Invoices?",
    "How many