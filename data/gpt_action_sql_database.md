# Guide: Building a GPT Action to Query a SQL Database

## Introduction

This guide walks you through the process of giving a GPT the ability to query a SQL database using a GPT Action. By the end, you'll have a system where users can ask questions in natural language and receive answers based on your live database.

### Prerequisites

*   A basic understanding of [GPT Actions](https://platform.openai.com/docs/actions).
*   A SQL database (PostgreSQL is used in this example).
*   A middleware application (custom-built or third-party) to act as a bridge between the GPT and your database.

### Value Proposition

This integration allows users to interact with complex data using simple language. Business users can get answers without writing SQL, and analysts can combine SQL queries with advanced AI analysis.

## Architecture Overview

Since SQL databases typically lack direct REST APIs, a middleware application is required. This middleware will:
1.  Receive SQL queries from the GPT via a REST API.
2.  Execute those queries against the database.
3.  Convert the results into a format the GPT can process (like a CSV file).
4.  Return the data to the GPT.

This guide focuses on a flexible design where the middleware accepts **arbitrary SQL queries**. This approach is easier to build and maintain, though it requires careful consideration of security and permissions.

## Step-by-Step Implementation

### Step 1: Configure the GPT's Instructions

The GPT needs context about your database to generate accurate SQL. Provide the schema directly in its instructions. Here is a template you can adapt:

```markdown
**Context**
You are a data analyst. Your job is to assist users with their business questions by analyzing the data contained in a PostgreSQL database.

## Database Schema

### Accounts Table
**Description:** Stores information about business accounts.

| Column Name  | Data Type      | Constraints                        | Description                             |
|--------------|----------------|------------------------------------|-----------------------------------------|
| account_id   | INT            | PRIMARY KEY, AUTO_INCREMENT, NOT NULL | Unique identifier for each account      |
| account_name | VARCHAR(255)   | NOT NULL                           | Name of the business account            |
| industry     | VARCHAR(255)   |                                    | Industry to which the business belongs  |
| created_at   | TIMESTAMP      | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Timestamp when the account was created  |

### Users Table
**Description:** Stores information about users associated with the accounts.

| Column Name  | Data Type      | Constraints                        | Description                             |
|--------------|----------------|------------------------------------|-----------------------------------------|
| user_id      | INT            | PRIMARY KEY, AUTO_INCREMENT, NOT NULL | Unique identifier for each user         |
| account_id   | INT            | NOT NULL, FOREIGN KEY (References Accounts(account_id)) | Foreign key referencing Accounts(account_id) |
| username     | VARCHAR(50)    | NOT NULL, UNIQUE                   | Username chosen by the user             |
| email        | VARCHAR(100)   | NOT NULL, UNIQUE                   | User's email address                    |
| role         | VARCHAR(50)    |                                    | Role of the user within the account     |
| created_at   | TIMESTAMP      | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Timestamp when the user was created     |

### Revenue Table
**Description:** Stores revenue data related to the accounts.

| Column Name  | Data Type      | Constraints                        | Description                             |
|--------------|----------------|------------------------------------|-----------------------------------------|
| revenue_id   | INT            | PRIMARY KEY, AUTO_INCREMENT, NOT NULL | Unique identifier for each revenue record |
| account_id   | INT            | NOT NULL, FOREIGN KEY (References Accounts(account_id)) | Foreign key referencing Accounts(account_id) |
| amount       | DECIMAL(10, 2) | NOT NULL                           | Revenue amount                          |
| revenue_date | DATE           | NOT NULL                           | Date when the revenue was recorded      |

**Instructions:**
1.  When the user asks a question, consider what data you need to answer it by consulting the database schema.
2.  Write a PostgreSQL-compatible query and submit it using the `databaseQuery` API method.
3.  Use the response data to answer the user's question.
4.  If necessary, use the Code Interpreter capability to perform additional analysis on the data.
```

**Note:** For dynamic or complex schemas, instruct the GPT to first query the database for schema information instead of hardcoding it.

### Step 2: Define the GPT Action API Schema

The GPT Action needs a well-defined interface to call your middleware. Define an OpenAPI schema for a `/query` endpoint. This schema tells the GPT how to structure its requests.

```yaml
openapi: 3.1.0
info:
  title: PostgreSQL API
  description: API for querying a PostgreSQL database
  version: 1.0.0
servers:
  - url: https://your-middleware-url.com/v1
    description: Middleware service
paths:
  /api/query:
    post:
      operationId: databaseQuery
      summary: Query a PostgreSQL database
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                q:
                  type: string
                  example: select * from users
      responses:
        "200":
          description: database records
          content:
            application/json:
              schema:
                type: object
                properties:
                  openaiFileResponse:
                    type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                          description: The name of the file.
                        mime_type:
                          type: string
                          description: The MIME type of the file.
                        content:
                          type: string
                          format: byte
                          description: The content of the file in base64 encoding.
        "400":
          description: Bad Request. Invalid input.
        "401":
          description: Unauthorized. Invalid or missing API key.
      security:
        - ApiKey: []
components:
  securitySchemes:
    ApiKey:
      type: apiKey
      in: header
      name: X-Api-Key
```

**Key Points:**
*   **Authentication:** This example uses a simple API key (`X-Api-Key`). For user-level permissions, implement OAuth. Your middleware would then need to map the authenticated user to specific database roles or permissions.
*   **Response Format:** The `200` response schema is structured to return a file via the `openaiFileResponse` field, which is required for the GPT to process CSV data.

### Step 3: Build the Middleware Logic

Your middleware application must handle the incoming request, execute the SQL, and format the response. Below is the core logic, broken into functions.

#### 3.1. Execute the Database Query

Use a database client library (like `psycopg2` for Python/PostgreSQL) to run the query received from the GPT.

```python
import psycopg2
import json

def execute_query(sql_string, db_connection_params):
    """
    Executes a SQL query against the PostgreSQL database.
    """
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(**db_connection_params)
        cur = conn.cursor()
        
        # Execute the query
        cur.execute(sql_string)
        
        # Fetch all results
        rows = cur.fetchall()
        # Get column names from the cursor description
        col_names = [desc[0] for desc in cur.description]
        
        # Convert rows to a list of dictionaries
        result = []
        for row in rows:
            result.append(dict(zip(col_names, row)))
        
        cur.close()
        return result
        
    except Exception as error:
        print(f"Database error: {error}")
        raise
    finally:
        if conn is not None:
            conn.close()

# Example usage (connection parameters would come from your environment)
db_params = {
    "host": "localhost",
    "database": "your_database",
    "user": "your_user",
    "password": "your_password"
}

# The 'q' parameter from the GPT's request
sql_from_gpt = "SELECT account_id, COUNT(*) as number_of_users FROM users GROUP BY account_id;"
query_results = execute_query(sql_from_gpt, db_params)
print(json.dumps(query_results, indent=2))
```

**Security Critical:** The database user your middleware uses should have **READ-ONLY** permissions. This prevents accidental or malicious data modification by AI-generated queries.

#### 3.2. Convert Results to a Base64-Encoded CSV

The GPT analyzes data most effectively from CSV files. Your middleware must convert the query results and encode them.

```python
import csv
import base64
import io
import json

def convert_to_csv_and_encode(data):
    """
    Converts a list of dictionaries to a base64-encoded CSV string.
    """
    if not data:
        # Handle empty result sets
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["message"])
        writer.writerow(["No data found for the query."])
        csv_string = output.getvalue()
    else:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        csv_string = output.getvalue()
    
    # Encode to base64
    csv_bytes = csv_string.encode('utf-8')
    base64_encoded = base64.b64encode(csv_bytes).decode('utf-8')
    
    return base64_encoded

# Example: Process the results from the previous step
encoded_csv_data = convert_to_csv_and_encode(query_results)
print("Base64 CSV Data (first 100 chars):", encoded_csv_data[:100])
```

#### 3.3. Structure the Final API Response

Format the response according to the `openaiFileResponse` schema required by GPT Actions.

```python
def create_api_response(encoded_csv_data, filename="query_results.csv"):
    """
    Creates the final JSON response for the GPT Action.
    """
    response = {
        "openaiFileResponse": [
            {
                "name": filename,
                "mime_type": "text/csv",
                "content": encoded_csv_data
            }
        ]
    }
    return response

# Create the final response object
api_response = create_api_response(encoded_csv_data)
print(json.dumps(api_response, indent=2))
```

### Step 4: Integrate and Test

1.  **Deploy Middleware:** Deploy your middleware code (e.g., as an Azure Function, AWS Lambda, or a server) and note its public URL.
2.  **Configure the GPT Action:**
    *   In the GPT editor, go to the **Configure** tab.
    *   Under **Actions**, click **Create new action**.
    *   Paste your OpenAPI schema from Step 2. The GPT will validate the schema and populate authentication details.
    *   Enter your middleware's URL and API key.
3.  **Enable Code Interpreter:** In the GPT editor's **Capabilities** section, ensure **Code Interpreter & Data Analysis** is enabled. This allows the GPT to perform calculations on the returned CSV data.
4.  **Test the Flow:** Ask your GPT a natural language question like, "What is the total revenue per industry?" The GPT should:
    *   Generate a SQL query (e.g., `SELECT industry, SUM(amount) FROM accounts JOIN revenue ON accounts.account_id = revenue.account_id GROUP BY industry`).
    *   Call your middleware's `/api/query` endpoint with that query.
    *   Receive the base64 CSV file, decode it, and use the data to formulate an answer.

## Security and Advanced Considerations

*   **User-Level Permissions:** For production use with multiple users, implement OAuth. Your middleware must:
    1.  Identify the user from the OAuth token.
    2.  Apply corresponding row-level or table-level security in the database (e.g., by setting a session variable like `SET app.user_id = '123';` and using it in row security policies).
*   **Dynamic Schema:** For complex permissions, don't hardcode the schema. Instead, instruct the GPT to first run a query like `SELECT table_name, column_name FROM information_schema.columns` to discover the schema visible to the current user.
*   **Query Validation:** Consider adding a lightweight validation layer in your middleware to block obviously dangerous queries (e.g., those containing `DROP`, `DELETE`, `INSERT`, `UPDATE` keywords) as a secondary safeguard beyond read-only database permissions.

## Conclusion

You have now built a pipeline that connects a GPT to a live SQL database. This pattern unlocks powerful natural language querying for your data. Remember to prioritize security by using read-only database credentials and considering user-level access control for sensitive data.

*Ready to extend this further? Consider adding endpoints for specific, high-use queries for more control, or integrating other data sources.*