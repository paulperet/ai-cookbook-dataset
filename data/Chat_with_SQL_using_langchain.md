# Chat with a SQL Database using Gemini and LangChain

## Overview
Interacting with SQL databases can be complex, but Large Language Models (LLMs) can simplify this process. In this guide, you will build a system that translates natural language questions into SQL queries, executes them against a database, and returns human-readable answers using Google's Gemini model and the LangChain framework.

## Prerequisites & Setup

First, install the required Python packages.

```bash
pip install -U -q "google-genai>=1.7.0" langchain langchain-community langchain-google-genai scikit-learn
```

Next, import the necessary modules.

```python
import sqlite3
import os

from langchain.chains import create_sql_query_chain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from google import genai
from IPython.display import Markdown
from sklearn.datasets import fetch_california_housing
```

## Step 1: Configure Your API Key

To use the Gemini API, you need a valid API key. Store your key in an environment variable named `GOOGLE_API_KEY`.

```python
# Replace 'YOUR_API_KEY' with your actual key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
```

## Step 2: Set Up the Database

You will create a local SQLite database using the California Housing dataset.

### 2.1 Load the Dataset
Load the dataset from `scikit-learn` and convert it into a pandas DataFrame.

```python
california_housing_bunch = fetch_california_housing(as_frame=True)
california_housing_df = california_housing_bunch.frame
```

### 2.2 Create the SQLite Database and Table
Create a connection to a new SQLite database file and write the DataFrame to a table named `housing`.

```python
# Connect to (or create) the SQLite database
conn = sqlite3.connect("mydatabase.db")

# Write the DataFrame to the 'housing' table
california_housing_df.to_sql("housing", conn, index=False)
```

### 2.3 Create a LangChain SQLDatabase Object
The `SQLDatabase` wrapper allows LangChain to interact with your database.

```python
db = SQLDatabase.from_uri("sqlite:///mydatabase.db")
```

You can inspect the database schema to confirm the setup.

```python
Markdown(db.get_table_info())
```

The output shows the table structure and a sample of the data:

```
CREATE TABLE housing (
	"MedInc" REAL,
	"HouseAge" REAL,
	"AveRooms" REAL,
	"AveBedrms" REAL,
	"Population" REAL,
	"AveOccup" REAL,
	"Latitude" REAL,
	"Longitude" REAL,
	"MedHouseVal" REAL
)

/*
3 rows from housing table:
MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude	MedHouseVal
8.3252	41.0	6.984126984126984	1.0238095238095237	322.0	2.5555555555555554	37.88	-122.23	4.526
8.3014	21.0	6.238137082601054	0.9718804920913884	2401.0	2.109841827768014	37.86	-122.22	3.585
7.2574	52.0	8.288135593220339	1.073446327683616	496.0	2.8022598870056497	37.85	-122.24	3.521
*/
```

## Step 3: Generate SQL Queries from Natural Language

Now, you will create a chain that uses an LLM to translate a user's question into a SQL query.

### 3.1 Initialize the LLM and Query Chain
Initialize the Gemini model and create a SQL query generation chain.

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
write_query_chain = create_sql_query_chain(llm, db)
```

The `create_sql_query_chain` function uses a default prompt tailored for SQLite. You can inspect this prompt:

```python
Markdown(write_query_chain.get_prompts()[0].template)
```

The prompt instructs the model to generate a syntactically correct SQLite query, execute it, and format the answer.

### 3.2 Test the Query Generation
Let's test the chain by asking a simple question.

```python
response = write_query_chain.invoke({"question": "What is the total population?"})
display(Markdown(response))
```

The chain outputs the generated SQL query:

```sql
SELECT sum("Population") FROM housing
```

You can manually execute this query to verify it works:

```python
db.run('SELECT SUM("Population") FROM housing')
```

The result is:
```
'[(29421840.0,)]'
```

The query is correct, but the output from the chain includes markdown formatting. You need to extract the raw SQL command before execution.

## Step 4: Validate and Extract the SQL Query

Create a validation step to extract a clean SQL command from the LLM's response.

### 4.1 Create a Validation Prompt and Chain
Define a prompt that instructs the model to extract only the valid SQL command.

```python
validate_prompt = PromptTemplate(
    input_variables=["not_formatted_query"],
    template="""
        You are going to receive a text that contains a SQL query. Extract that query.
        Make sure that it is a valid SQL command that can be passed directly to the Database.
        Avoid using Markdown for this task.
        Text: {not_formatted_query}
    """
)
```

Chain the query generation, validation, and output parsing steps together.

```python
validate_chain = write_query_chain | validate_prompt | llm | StrOutputParser()
```

### 4.2 Test the Validation Chain
Test the new chain with the same question.

```python
validate_chain.invoke({"question": "What is the total population?"})
```

The output is now a clean SQL string:
```
'SELECT sum("Population") FROM housing'
```

## Step 5: Automate Query Execution

Use LangChain's `QuerySQLDataBaseTool` to automatically execute the validated SQL query against the database.

```python
execute_query = QuerySQLDataBaseTool(db=db)
execute_chain = validate_chain | execute_query
```

Test the execution chain:

```python
execute_chain.invoke({"question": "What is the total population?"})
```

The tool executes the query and returns the result:
```
'[(29421840.0,)]'
```

## Step 6: Generate a Natural Language Answer

The final step is to format the raw SQL result into a natural language answer.

### 6.1 Create an Answer Generation Prompt
Define a prompt that takes the original question, the generated SQL, and the query result to produce a final answer.

```python
answer_prompt = PromptTemplate.from_template("""
    You are going to receive a original user question, generated SQL query, and result of said query. You should use this information to answer the original question. Use only information provided to you.

    Original Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)
```

### 6.2 Assemble the Final Answer Chain
Create a chain that:
1.  Passes through the original question.
2.  Generates and validates the SQL query.
3.  Executes the query.
4.  Formats the final answer.

```python
answer_chain = (
    RunnablePassthrough.assign(query=validate_chain).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt | llm | StrOutputParser()
)
```

### 6.3 Test the Complete Pipeline
Ask the same question to the complete system.

```python
answer_chain.invoke({"question": "What is the total population?"})
```

The system now provides a clear, human-readable answer:
```
'The total population is 29,421,840.'
```

## Conclusion

You have successfully built a pipeline that:
1.  Accepts a natural language question.
2.  Uses an LLM (Gemini) to generate a corresponding SQL query.
3.  Validates and executes the query against a SQLite database.
4.  Uses another LLM call to format the result into a natural language answer.

This system demonstrates a core Retrieval-Augmented Generation (RAG) pattern for structured data. You can extend this by asking more complex questions of your database or integrating it into a larger application.