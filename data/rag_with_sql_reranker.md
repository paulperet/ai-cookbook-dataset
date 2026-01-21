# Building a SQL-Backed RAG System with Jina Reranker v2

_Authored by: [Scott Martens](https://github.com/scott-martens) @ [Jina AI](https://jina.ai)_

This guide demonstrates how to build a Retrieval Augmented Generation (RAG) system that retrieves information from a SQL database instead of a traditional document store. You will use Jina Reranker v2 to intelligently select relevant database tables and Mistral 7B Instruct to generate SQL queries and natural language answers.

## How It Works

1.  **Table Definition Storage:** SQL table definitions (the `CREATE` statements) are extracted and stored. This example uses a pre-defined list.
2.  **User Query:** A user submits a question in natural language.
3.  **Table Reranking:** [Jina Reranker v2](https://jina.ai/reranker/) scores and ranks all table definitions based on their relevance to the query.
4.  **SQL Generation:** The top three table definitions and the user query are sent to [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) via the Hugging Face Inference API with a prompt to generate the appropriate SQL query.
5.  **Query Execution:** The generated SQL is executed against the SQLite database.
6.  **Answer Synthesis:** The SQL results, original query, and SQL query are sent back to Mistral Instruct with a prompt to formulate a concise, natural language answer for the user.

## Prerequisites

*   A **Hugging Face account** and an **access token** with at least `READ` permissions. Get your token from your [Hugging Face settings](https://huggingface.co/settings/tokens).
*   **Python 3.8+** (This guide uses Python 3.11).
*   For optimal performance, especially when running the Jina Reranker model locally, a **GPU (CUDA-enabled)** is recommended. A CPU will work but will be slower.
*   **SQLite** is required. It's built into Python, but the command-line tool may need to be installed separately on your system.

## Step 1: Environment Setup

### 1.1 Install Required Python Packages

Run the following command to install all necessary dependencies:

```bash
pip install -qU transformers einops llama-index llama-index-postprocessor-jinaai-rerank llama-index-llms-huggingface "huggingface_hub[inference]"
```

### 1.2 Download the Example Database

We'll use a sample video game sales database. Download the SQLite file to your working directory.

```bash
wget https://github.com/bbrumm/databasestar/raw/main/sample_databases/sample_db_videogames/sqlite/videogames.db
```

If `wget` is unavailable, manually download the file from the link above and place it in your project folder.

### 1.3 Load the Jina Reranker v2 Model

Load the `jina-reranker-v2-base-multilingual` model locally. Ensure you have sufficient GPU memory if using a GPU.

```python
from transformers import AutoModelForSequenceClassification

reranker_model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="auto",
    trust_remote_code=True,
)

# Move model to GPU for faster inference, or use 'cpu'
reranker_model.to('cuda')
reranker_model.eval()
```

### 1.4 Configure the Mistral LLM Connection

Set up the connection to the Mistral 7B Instruct model via the Hugging Face Inference API.

First, securely input your Hugging Face token:

```python
import getpass

print("Paste your Hugging Face access token here: ")
hf_token = getpass.getpass()
```

Next, create the LLM interface object using LlamaIndex:

```python
from llama_index.llms.huggingface import HuggingFaceInferenceAPI

mistral_llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.1", token=hf_token
)
```

## Step 2: Prepare the Table Definitions

The `CREATE TABLE` statements for our sample database have been pre-extracted. Store them in a list.

```python
table_declarations = [
    'CREATE TABLE platform (\n\tid INTEGER PRIMARY KEY,\n\tplatform_name TEXT DEFAULT NULL\n);',
    'CREATE TABLE genre (\n\tid INTEGER PRIMARY KEY,\n\tgenre_name TEXT DEFAULT NULL\n);',
    'CREATE TABLE publisher (\n\tid INTEGER PRIMARY KEY,\n\tpublisher_name TEXT DEFAULT NULL\n);',
    'CREATE TABLE region (\n\tid INTEGER PRIMARY KEY,\n\tregion_name TEXT DEFAULT NULL\n);',
    'CREATE TABLE game (\n\tid INTEGER PRIMARY KEY,\n\tgenre_id INTEGER,\n\tgame_name TEXT DEFAULT NULL,\n\tCONSTRAINT fk_gm_gen FOREIGN KEY (genre_id) REFERENCES genre(id)\n);',
    'CREATE TABLE game_publisher (\n\tid INTEGER PRIMARY KEY,\n\tgame_id INTEGER DEFAULT NULL,\n\tpublisher_id INTEGER DEFAULT NULL,\n\tCONSTRAINT fk_gpu_gam FOREIGN KEY (game_id) REFERENCES game(id),\n\tCONSTRAINT fk_gpu_pub FOREIGN KEY (publisher_id) REFERENCES publisher(id)\n);',
    'CREATE TABLE game_platform (\n\tid INTEGER PRIMARY KEY,\n\tgame_publisher_id INTEGER DEFAULT NULL,\n\tplatform_id INTEGER DEFAULT NULL,\n\trelease_year INTEGER DEFAULT NULL,\n\tCONSTRAINT fk_gpl_gp FOREIGN KEY (game_publisher_id) REFERENCES game_publisher(id),\n\tCONSTRAINT fk_gpl_pla FOREIGN KEY (platform_id) REFERENCES platform(id)\n);',
    'CREATE TABLE region_sales (\n\tregion_id INTEGER DEFAULT NULL,\n\tgame_platform_id INTEGER DEFAULT NULL,\n\tnum_sales REAL,\n   CONSTRAINT fk_rs_gp FOREIGN KEY (game_platform_id) REFERENCES game_platform(id),\n\tCONSTRAINT fk_rs_reg FOREIGN KEY (region_id) REFERENCES region(id)\n);'
]
```

## Step 3: Rank Tables with Jina Reranker v2

Now, create a function that uses the Jina Reranker model to score and sort table definitions based on a user's query.

```python
from typing import List, Tuple

def rank_tables(query: str, table_specs: List[str], top_n: int = 0) -> List[Tuple[float, str]]:
    """
    Scores and sorts table definitions by relevance to the query.
    Returns the top N results, or all if top_n is 0.
    """
    # Create query-table pairs for the reranker
    pairs = [[query, table_spec] for table_spec in table_specs]
    # Get relevance scores
    scores = reranker_model.compute_score(pairs)
    # Pair scores with their corresponding table definition
    scored_tables = [(score, table_spec) for score, table_spec in zip(scores, table_specs)]
    # Sort by score, highest first
    scored_tables.sort(key=lambda x: x[0], reverse=True)
    # Return the top N results
    if top_n and top_n < len(scored_tables):
        return scored_tables[0:top_n]
    return scored_tables
```

### 3.1 Test the Reranker

Let's test the function with a sample query.

```python
user_query = "Identify the top 10 platforms by total sales."
ranked_tables = rank_tables(user_query, table_declarations, top_n=3)
ranked_tables
```

The output should list the three most relevant tables (e.g., `region_sales`, `platform`, `game_platform`) along with their relevance scores.

## Step 4: Generate SQL with Mistral Instruct

### 4.1 Create the SQL Generation Prompt Template

We'll use a LlamaIndex `PromptTemplate` to structure the instruction for the LLM.

```python
from llama_index.core import PromptTemplate

make_sql_prompt_tmpl_text = """
Generate a SQL query to answer the following question from the user:
\"{query_str}\"

The SQL query should use only tables with the following SQL definitions:

Table 1:
{table_1}

Table 2:
{table_2}

Table 3:
{table_3}

Make sure you ONLY output an SQL query and no explanation.
"""
make_sql_prompt_tmpl = PromptTemplate(make_sql_prompt_tmpl_text)
```

### 4.2 Populate the Prompt and Generate SQL

Fill the template with the user's query and the top three table definitions, then send it to Mistral.

```python
# Format the prompt with the actual data
make_sql_prompt = make_sql_prompt_tmpl.format(
    query_str=user_query,
    table_1=ranked_tables[0][1],
    table_2=ranked_tables[1][1],
    table_3=ranked_tables[2][1]
)

# Generate the SQL query
response = mistral_llm.complete(make_sql_prompt)
sql_query = str(response)
print("Generated SQL Query:")
print(sql_query)
```

## Step 5: Execute the SQL Query

Use Python's built-in `sqlite3` library to run the generated query against the database.

```python
import sqlite3

# Connect to the database and execute the query
con = sqlite3.connect("videogames.db")
cur = con.cursor()
sql_response = cur.execute(sql_query).fetchall()

# Inspect the raw results
print("SQL Execution Results:")
print(sql_response)
```

## Step 6: Synthesize a Natural Language Answer

### 6.1 Create the Answer Synthesis Prompt

Define a second prompt template that asks the LLM to explain the SQL results in plain language.

```python
rag_prompt_tmpl_str = """
Use the information in the JSON table to answer the following user query.
Do not explain anything, just answer concisely. Use natural language in your
answer, not computer formatting.

USER QUERY: {query_str}

JSON table:
{json_table}

This table was generated by the following SQL query:
{sql_query}

Answer ONLY using the information in the table and the SQL query, and if the
table does not provide the information to answer the question, answer
"No Information".
"""
rag_prompt_tmpl = PromptTemplate(rag_prompt_tmpl_str)
```

### 6.2 Format and Send the Final Prompt

Convert the SQL results to JSON, populate the prompt, and get the final answer.

```python
import json

# Format the final prompt
rag_prompt = rag_prompt_tmpl.format(
    query_str=user_query,
    json_table=json.dumps(sql_response),
    sql_query=sql_query
)

# Generate the final answer
rag_response = mistral_llm.complete(rag_prompt)
print("Final Answer:")
print(str(rag_response))
```

## Step 7: Integrate Everything into a Single Function

Let's wrap the entire workflow into a robust, reusable function with basic error handling.

```python
def answer_sql(user_query: str) -> str:
    """
    End-to-end function: takes a natural language query and returns a natural language answer.
    """
    # 1. Rank relevant tables
    try:
        ranked_tables = rank_tables(user_query, table_declarations, top_n=3)
    except Exception as e:
        print(f"Table ranking failed for query: {user_query}")
        raise e

    # 2. Generate the SQL prompt and query
    make_sql_prompt = make_sql_prompt_tmpl.format(
        query_str=user_query,
        table_1=ranked_tables[0][1],
        table_2=ranked_tables[1][1],
        table_3=ranked_tables[2][1]
    )
    try:
        response = mistral_llm.complete(make_sql_prompt)
    except Exception as e:
        print(f"SQL generation failed. Prompt:\n{make_sql_prompt}")
        raise e

    # Clean up the generated SQL (remove escape characters)
    sql_query = str(response).replace("\\", "")

    # 3. Execute the SQL query
    try:
        con = sqlite3.connect("videogames.db")
        sql_response = con.cursor().execute(sql_query).fetchall()
        con.close()
    except Exception as e:
        print(f"SQL execution failed. Query:\n{sql_query}")
        raise e

    # 4. Generate the final answer
    rag_prompt = rag_prompt_tmpl.format(
        query_str=user_query,
        json_table=json.dumps(sql_response),
        sql_query=sql_query
    )
    try:
        rag_response = mistral_llm.complete(rag_prompt)
        return str(rag_response)
    except Exception as e:
        print(f"Answer generation failed. Prompt:\n{rag_prompt}")
        raise e
```

## Step 8: Test the Complete System

Now you can test the integrated pipeline with various queries.

```python
print("Query 1: Identify the top 10 platforms by total sales.")
print(answer_sql("Identify the top 10 platforms by total sales."))
print("\n" + "-"*50 + "\n")

print("Query 2: Summarize sales by region.")
print(answer_sql("Summarize sales by region."))
print("\n" + "-"*50 + "\n")

print("Query 3: List the publisher with the largest number of published games.")
print(answer_sql("List the publisher with the largest number of published games."))
print("\n" + "-"*50 + "\n")

print("Query 4: What is the most popular game genre on the Wii platform?")
print(answer_sql("What is the most popular game genre on the Wii platform?"))
```

Feel free to experiment with your own questions about the video game sales data.

## Conclusion and Next Steps

You have successfully built a functional SQL-backed RAG system. This example illustrates how the RAG pattern can be extended to structured data sources, significantly broadening its application scope.

**Key Points and Considerations for Production:**

*   **Scalability:** This example uses a simple list of table definitions. A real-world application with hundreds of tables would require a more sophisticated retrieval system, potentially combining embedding-based vector search with the reranker.
*   **Robustness:** The generated SQL is not guaranteed to be correct or executable. Production systems need comprehensive validation, error handling, and possibly a fallback mechanism.
*   **Context Limits:** We assumed three tables are sufficient. Complex queries may require more context or a different prompting strategy.
*   **Model Specialization:** Using an LLM specifically fine-tuned for SQL generation (like SQLCoder) could improve accuracy.

This tutorial provides a foundation. You can enhance it by integrating better retrieval methods, implementing SQL validation, and experimenting with different LLMs to suit your specific use case and database schema.