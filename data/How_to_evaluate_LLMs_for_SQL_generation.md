# Evaluating LLMs for SQL Generation: A Step-by-Step Guide

Large Language Models (LLMs) are inherently non-deterministic, making them creative but challenging to deploy consistently in production. This guide demonstrates a systematic evaluation framework for LLM applications, focusing on SQL generation from natural language. You'll learn to implement unit tests, evaluation metrics, and runbook documentation to track performance over time.

## Prerequisites

### Install Required Libraries
First, install the necessary Python packages:

```bash
pip install openai datasets pandas pydantic matplotlib python-dotenv numpy tqdm
```

### Import Libraries and Load Data
Import the required modules and load the SQL dataset:

```python
from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import pydantic
import os
import sqlite3
from sqlite3 import Error
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from IPython.display import HTML, display

# Load environment variables from .env file
load_dotenv()

GPT_MODEL = 'gpt-4o'
dataset = load_dataset("b-mc2/sql-create-context")

print(dataset['train'].num_rows, "rows")
```

This loads the [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset from HuggingFace, which contains 78,577 examples of natural language questions paired with corresponding SQL queries and table schemas.

### Examine the Dataset Structure
Let's look at the dataset format:

```python
sql_df = dataset['train'].to_pandas()
sql_df.head()
```

Each row contains three key components:
1. **Question**: Natural language query
2. **Context**: CREATE TABLE statement defining the schema
3. **Answer**: SELECT query that answers the question

## Step 1: Define the Expected Response Structure

We'll use Pydantic to define the expected JSON structure for LLM responses:

```python
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Expected structure for LLM responses.
    
    The LLM should return a JSON object with 'create' and 'select' fields
    containing SQL statements.
    """
    create: str
    select: str
```

## Step 2: Create the SQL Generation Function

Define a function that prompts the LLM to generate SQL from natural language:

```python
system_prompt = """Translate this natural language request into a JSON
object containing two SQL queries. The first query should be a CREATE 
statement for a table answering the user's request, while the second
should be a SELECT query answering their question."""

client = OpenAI()

def get_response(system_prompt, user_message, model=GPT_MODEL):
    """Get SQL generation from LLM with structured JSON response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    response = client.beta.chat.completions.parse(
        model=GPT_MODEL,
        messages=messages,
        response_format=LLMResponse,
    )
    return response.choices[0].message.content
```

Test the function with a sample question:

```python
question = sql_df.iloc[0]['question']
content = get_response(system_prompt, question)
print("Question:", question)
print("Answer:", content)
```

## Step 3: Implement Unit Tests

### Test 1: Validate JSON Schema
Create a test to verify the LLM returns properly formatted JSON:

```python
def test_valid_schema(content):
    """Test whether content can be parsed into our Pydantic model."""
    try:
        LLMResponse.model_validate_json(content)
        return True
    except pydantic.ValidationError as exc:
        print(f"ERROR: Invalid schema: {exc}")
        return False
```

Test with valid and invalid responses:

```python
# Test with valid response
test_valid_schema(content)

# Test with invalid response
failing_query = 'CREATE departments, select * from departments'
test_valid_schema(failing_query)
```

### Test 2: Validate SQL Syntax
Create SQLite helper functions to test SQL execution:

```python
def create_connection(db_file=":memory:"):
    """Create a database connection to a SQLite database."""
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
        return None

def close_connection(conn):
    """Close a database connection."""
    try:
        conn.close()
    except Error as e:
        print(e)

def test_select(conn, cursor, select, should_log=True):
    """Test that a SQLite SELECT query executes successfully."""
    try:
        if should_log:
            print(f"Testing select query: {select}")
        cursor.execute(select)
        record = cursor.fetchall()
        if should_log:
            print(f"Result of query: {record}")
        return True
    except sqlite3.Error as error:
        if should_log:
            print("Error while executing select query:", error)
        return False

def test_create(conn, cursor, create, should_log=True):
    """Test that a SQLite CREATE query executes successfully."""
    try:
        if should_log:
            print(f"Testing create query: {create}")
        cursor.execute(create)
        conn.commit()
        return True
    except sqlite3.Error as error:
        if should_log:
            print("Error while creating the SQLite table:", error)
        return False

def test_llm_sql(llm_response, should_log=True):
    """Run a suite of SQLite tests on LLM-generated SQL."""
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        create_response = test_create(conn, cursor, llm_response.create, should_log=should_log)
        select_response = test_select(conn, cursor, llm_response.select, should_log=should_log)
        
        if conn:
            close_connection(conn)
        
        return create_response and select_response
    except sqlite3.Error as error:
        if should_log:
            print("Error while creating a sqlite table", error)
        return False
```

Test the SQL validation:

```python
# Parse the LLM response
test_query = LLMResponse.model_validate_json(content)

# View the generated SQL
print(f"CREATE SQL is: {test_query.create}")
print(f"SELECT SQL is: {test_query.select}")

# Test SQL execution
test_llm_sql(test_query)
```

## Step 4: Implement Relevance Evaluation

Use an LLM to evaluate whether generated SQL actually answers the user's question, adapting the G-Eval approach:

```python
EVALUATION_MODEL = "gpt-4o-mini"

EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Request:

{request}

Queries:

{queries}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Relevance criteria
RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - review of how relevant the produced SQL queries are to the original question. \
The queries should contain all points highlighted in the user's request. \
Annotators were instructed to penalize queries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the request and the queries carefully.
2. Compare the queries to the request document and identify the main points of the request.
3. Assess how well the queries cover the main points of the request, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

def get_geval_score(criteria: str, steps: str, request: str, queries: str, metric_name: str):
    """Use an LLM to evaluate observations against specified criteria."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        request=request,
        queries=queries,
        metric_name=metric_name,
    )
    response = client.chat.completions.create(
        model=EVALUATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content
```

Test the evaluation on sample data:

```python
evaluation_results = []

for x, y in sql_df.head(3).iterrows():
    score = get_geval_score(
        RELEVANCY_SCORE_CRITERIA,
        RELEVANCY_SCORE_STEPS,
        y['question'],
        y['context'] + '\n' + y['answer'],
        'relevancy'
    )
    evaluation_results.append((y['question'], y['context'] + '\n' + y['answer'], score))

for result in evaluation_results:
    print(f"User Question: {result[0]}")
    print(f"CREATE SQL: {result[1].splitlines()[0]}")
    print(f"SELECT SQL: {result[1].splitlines()[1]}")
    print(f"Relevance Score: {result[2]}")
    print("*" * 20)
```

## Step 5: Build the Complete Test Framework

Create a comprehensive testing function that combines all evaluations:

```python
def execute_unit_tests(input_df, output_list, system_prompt):
    """Run unit tests and evaluations on a dataframe of questions."""
    
    for x, y in tqdm(input_df.iterrows(), total=len(input_df)):
        model_response = get_response(system_prompt, y['question'])
        
        format_valid = test_valid_schema(model_response)
        
        try:
            test_query = LLMResponse.model_validate_json(model_response)
            sql_valid = test_llm_sql(test_query, should_log=False)
        except:
            sql_valid = False
        
        # Store results
        output_list.append({
            'question': y['question'],
            'response': model_response,
            'format_valid': format_valid,
            'sql_valid': sql_valid
        })
```

## Step 6: Compare Different Prompts

Now you can compare different system prompts to see which performs better. Create two versions:

```python
# Original prompt
system_prompt_v1 = """Translate this natural language request into a JSON
object containing two SQL queries. The first query should be a CREATE 
statement for a table answering the user's request, while the second
should be a SELECT query answering their question."""

# Enhanced prompt with additional guidance
system_prompt_v2 = """Translate this natural language request into a JSON
object containing two SQL queries. The first query should be a CREATE 
statement for a table answering the user's request, while the second
should be a SELECT query answering their question.

IMPORTANT: Ensure the SELECT query only uses columns that exist in the 
CREATE statement, and that the WHERE clause conditions are appropriate 
for the data types defined."""

# Run tests with both prompts
results_v1 = []
results_v2 = []

execute_unit_tests(sql_df.head(100), results_v1, system_prompt_v1)
execute_unit_tests(sql_df.head(100), results_v2, system_prompt_v2)
```

## Step 7: Analyze and Report Results

Calculate and compare performance metrics:

```python
def calculate_metrics(results):
    """Calculate key performance metrics from test results."""
    total = len(results)
    format_success = sum(1 for r in results if r['format_valid'])
    sql_success = sum(1 for r in results if r['sql_valid'])
    
    return {
        'format_success_rate': format_success / total * 100,
        'sql_success_rate': sql_success / total * 100,
        'overall_success_rate': sum(1 for r in results if r['format_valid'] and r['sql_valid']) / total * 100
    }

metrics_v1 = calculate_metrics(results_v1)
metrics_v2 = calculate_metrics(results_v2)

print("Prompt V1 Metrics:", metrics_v1)
print("Prompt V2 Metrics:", metrics_v2)
```

Create a visual comparison:

```python
labels = ['Format Success', 'SQL Success', 'Overall Success']
v1_scores = [metrics_v1['format_success_rate'], metrics_v1['sql_success_rate'], metrics_v1['overall_success_rate']]
v2_scores = [metrics_v2['format_success_rate'], metrics_v2['sql_success_rate'], metrics_v2['overall_success_rate']]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, v1_scores, width, label='Prompt V1')
rects2 = ax.bar(x + width/2, v2_scores, width, label='Prompt V2')

ax.set_ylabel('Success Rate (%)')
ax.set_title('Prompt Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
```

## Conclusion

This framework provides a systematic approach to evaluating LLM performance for SQL generation. By implementing:

1. **Unit tests** for JSON formatting and SQL syntax
2. **LLM-based evaluation** for relevance scoring
3. **Comparative analysis** between different prompts

You can objectively measure improvements and make data-driven decisions about which prompts and configurations work best for your use case. This methodology extends beyond SQL generation to any LLM application where you need consistent, measurable performance.