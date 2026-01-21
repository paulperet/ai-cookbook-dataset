# Building a GPT Action for Google BigQuery

## Introduction

This guide walks you through creating a GPT Action that connects to **Google BigQuery**, Google Cloud's enterprise data warehouse. The Action enables users to ask natural language questions about their data, which are then translated into SQL queries. The Action returns the SQL statement for execution, providing a powerful interface for data exploration.

**Key Value**: Enables natural language interaction with BigQuery datasets, making data analysis accessible to both technical and non-technical users.

**Example Use Cases**:
- Data scientists performing exploratory data analysis
- Business users querying transactional data
- Teams investigating data anomalies or trends

## Prerequisites

Before you begin, ensure you have:

1. **A Google Cloud Platform (GCP) project** with BigQuery enabled
2. **A BigQuery dataset** within your GCP project
3. **Authentication credentials** for a user/service account with access to the dataset
4. **OpenAI GPTs access** to create Custom GPTs with Actions

## Step 1: Set Up Authentication in Google Cloud

First, configure OAuth 2.0 credentials in the Google Cloud Console:

1. Navigate to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth client ID**
3. Select **Web application** as the application type
4. Note down your **Client ID** and **Client Secret** securely
5. You'll add the callback URL later in Step 4

## Step 2: Configure the Custom GPT Instructions

Create a new Custom GPT and paste the following instructions into the **Instructions** panel:

```
**Context**: You are an expert at writing BigQuery SQL queries. A user is going to ask you a question.

**Instructions**:
1. No matter the user's question, start by running `runQuery` operation using this query: "SELECT column_name, table_name, data_type, description FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`"
   -- Assume project = "<insert your default project here>", dataset = "<insert your default dataset here>", unless the user provides different values
   -- Remember to include useLegacySql:false in the json output
2. Convert the user's question into a SQL statement that leverages the step above and run the `runQuery` operation on that SQL statement to confirm the query works. Add a limit of 100 rows
3. Now remove the limit of 100 rows and return back the query for the user to see

**Additional Notes**: If the user says "Let's get started", explain that the user can provide a project or dataset, along with a question they want answered. If the user has no ideas, suggest that we have a sample flights dataset they can query - ask if they want you to query that
```

**Important**: Replace `<insert your default project here>` and `<insert your default dataset here>` with your actual GCP project ID and dataset name.

## Step 3: Add the OpenAPI Schema

In the same Custom GPT, navigate to the **Actions** panel and paste the following OpenAPI schema:

```yaml
openapi: 3.1.0
info:
  title: BigQuery API
  description: API for querying a BigQuery table.
  version: 1.0.0
servers:
  - url: https://bigquery.googleapis.com/bigquery/v2
    description: Google BigQuery API server
paths:
  /projects/{projectId}/queries:
    post:
      operationId: runQuery
      summary: Executes a query on a specified BigQuery table.
      description: Submits a query to BigQuery and returns the results.
      x-openai-isConsequential: false
      parameters:
        - name: projectId
          in: path
          required: true
          description: The ID of the Google Cloud project.
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  description: The SQL query string.
                useLegacySql:
                  type: boolean
                  description: Whether to use legacy SQL.
                  default: false
      responses:
        '200':
          description: Successful query execution.
          content:
            application/json:
              schema:
                type: object
                properties:
                  kind:
                    type: string
                    example: "bigquery#queryResponse"
                  schema:
                    type: object
                    description: The schema of the results.
                  jobReference:
                    type: object
                    properties:
                      projectId:
                        type: string
                      jobId:
                        type: string
                  rows:
                    type: array
                    items:
                      type: object
                      properties:
                        f:
                          type: array
                          items:
                            type: object
                              properties:
                                v:
                                  type: string
                  totalRows:
                    type: string
                    description: Total number of rows in the query result.
                  pageToken:
                    type: string
                    description: Token for pagination of query results.
        '400':
          description: Bad request. The request was invalid.
        '401':
          description: Unauthorized. Authentication is required.
        '403':
          description: Forbidden. The request is not allowed.
        '404':
          description: Not found. The specified resource was not found.
        '500':
          description: Internal server error. An error occurred while processing the request.
```

## Step 4: Configure OAuth Authentication

In the Custom GPT's **Authentication** section:

1. Select **OAuth** as the authentication type
2. Enter the following details:
   - **Client ID**: Your Google Cloud OAuth Client ID
   - **Client Secret**: Your Google Cloud OAuth Client Secret
   - **Authorization URL**: `https://accounts.google.com/o/oauth2/auth`
   - **Token URL**: `https://oauth2.googleapis.com/token`
   - **Scope**: `https://www.googleapis.com/auth/bigquery`
   - **Token Exchange Method**: POST (default)

3. Copy the **Callback URL** provided by ChatGPT
4. Return to Google Cloud Console and add this callback URL to your OAuth client's **Authorized redirect URIs**

## Step 5: Test Your GPT Action

Your GPT Action is now ready for testing:

1. Save your Custom GPT configuration
2. Start a conversation with your GPT
3. Begin with "Let's get started" to see the welcome message
4. Provide a project and dataset (or use your defaults)
5. Ask a natural language question about your data

**Example interaction**:
- User: "Show me the top 10 customers by total purchases"
- GPT Action: 
  1. First queries the INFORMATION_SCHEMA to understand table structure
  2. Generates SQL: `SELECT customer_id, SUM(purchase_amount) as total_purchases FROM sales GROUP BY customer_id ORDER BY total_purchases DESC LIMIT 10`
  3. Returns the final SQL without the LIMIT clause for execution

## Troubleshooting

### Common Issues and Solutions

**Callback URL Error**:
- Ensure you've added the exact callback URL from ChatGPT to your Google Cloud OAuth client's "Authorized redirect URIs"
- URLs are case-sensitive and must match exactly

**Wrong Project/Dataset Being Queried**:
- Update the instructions to explicitly specify your default project and dataset
- Or modify instructions to require users to provide project/dataset details before querying

**Authentication Failures**:
- Verify your OAuth credentials are correctly entered in ChatGPT
- Ensure the user/service account has BigQuery permissions
- Check that the BigQuery API is enabled in your GCP project

**Query Errors**:
- The Action first tests queries with a LIMIT 100 to catch syntax errors
- Review the generated SQL for table/column name mismatches
- Ensure your dataset exists in the specified project

## Next Steps

This Action returns SQL statements rather than query results. For a complete solution that executes queries and returns data:

1. Consider building middleware to execute the SQL and return CSV/JSON results
2. Implement query validation and safety checks
3. Add support for more complex queries and joins
4. Implement query caching for performance

## Resources

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [OpenAI GPT Actions Documentation](https://platform.openai.com/docs/actions)
- [OAuth 2.0 for Google APIs](https://developers.google.com/identity/protocols/oauth2)

---

*Note: This guide provides the foundation for a BigQuery GPT Action. For production use, consider adding error handling, query optimization, and result formatting based on your specific requirements.*