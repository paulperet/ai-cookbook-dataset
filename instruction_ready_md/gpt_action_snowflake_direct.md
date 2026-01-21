# Guide: Building a GPT Action for Snowflake

## Introduction
This guide walks you through building a custom GPT Action that connects directly to a Snowflake Data Warehouse. The Action allows users to ask questions in natural language, which are then translated into SQL queries, executed against your Snowflake schema, and results returned.

**Key Capability:** The Action automatically inspects your database schema, crafts a relevant SQL query, validates it with a limited run, and then provides the final query for execution.

> **Note:** This implementation returns a SQL `ResultSet` statement. For production use where large result sets exceed GPT Action payload limits, consider the [Snowflake Middleware cookbook](../gpt_actions_library/gpt_action_snowflake_middleware) which handles CSV output.

### Prerequisites
Before you begin, ensure you have:
* A provisioned Snowflake Data Warehouse.
* A user role with access to the necessary databases, schemas, and tables.
* Familiarity with [GPT Actions](https://platform.openai.com/docs/actions) and the [Snowflake SQL API](https://docs.snowflake.com/en/developer-guide/sql-api/intro).

---

## Part 1: Configure the Custom GPT

### Step 1.1: Set GPT Instructions
In your Custom GPT's configuration panel, navigate to the **Instructions** section and paste the following. Replace the placeholders (`<insert your default warehouse here>`, `<insert your default database here>`, `<your_role>`) with your Snowflake details.

```text
**Context**: You are an expert at writing Snowflake SQL queries. A user is going to ask you a question.

**Instructions**:
1. No matter the user's question, start by running `runQuery` operation using this query: "SELECT column_name, table_name, data_type, comment FROM {database}.INFORMATION_SCHEMA.COLUMNS"
-- Assume warehouse = "<insert your default warehouse here>", database = "<insert your default database here>", unless the user provides different values
2. Convert the user's question into a SQL statement that leverages the step above and run the `runQuery` operation on that SQL statement to confirm the query works. Add a limit of 100 rows
3. Now remove the limit of 100 rows and return back the query for the user to see
4. Use the <your_role> role when querying Snowflake
5. Run each step in sequence. Explain what you are doing in a few sentences, run the action, and then explain what you learned. This will help the user understand the reason behind your workflow.

**Additional Notes**: If the user says "Let's get started", explain that the user can provide a project or dataset, along with a question they want answered. If the user has no ideas, suggest that we have a sample flights dataset they can query - ask if they want you to query that
```

These instructions guide the GPT to first understand your database schema, then iteratively build and test a query based on the user's question.

### Step 1.2: Configure the OpenAPI Schema
In the **Actions** panel of your Custom GPT, you need to define the interface to Snowflake. Paste the following OpenAPI schema. You **must** update the `servers.url` field with your specific Snowflake account URL in the format: `https://<orgname>-<account_name>.snowflakecomputing.com/api/v2`.

```yaml
openapi: 3.1.0
info:
  title: Snowflake Statements API
  version: 1.0.0
  description: API for executing statements in Snowflake with specific warehouse and role settings.
servers:
  - url: 'https://<orgname>-<account_name>.snowflakecomputing.com/api/v2'
paths:
  /statements:
    post:
      summary: Execute a SQL statement in Snowflake
      description: This endpoint allows users to execute a SQL statement in Snowflake, specifying the warehouse and roles to use.
      operationId: runQuery
      tags:
        - Statements
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                warehouse:
                  type: string
                  description: The name of the Snowflake warehouse to use for the statement execution.
                role:
                  type: string
                  description: The Snowflake role to assume for the statement execution.
                statement:
                  type: string
                  description: The SQL statement to execute.
              required:
                - warehouse
                - role
                - statement
      responses:
        '200':
          description: Successful execution of the SQL statement.
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  data:
                    type: object
                    additionalProperties: true
        '400':
          description: Bad request, e.g., invalid SQL statement or missing parameters.
        '401':
          description: Authentication error, invalid API credentials.
        '500':
          description: Internal server error.
```

This schema defines the `runQuery` operation that the GPT will call to execute SQL.

---

## Part 2: Configure Snowflake Integration

### Step 2.1: Configure IP Whitelisting for ChatGPT
If your Snowflake account uses network policies, you must whitelist ChatGPT's IP egress ranges.

1.  Review the current IP ranges listed at: [https://platform.openai.com/docs/actions/production/ip-egress-ranges](https://platform.openai.com/docs/actions/production/ip-egress-ranges).
2.  In a Snowflake worksheet, create a network rule and policy using those IPs.

```sql
-- Example: Create a network rule with IPs from the OpenAI list
CREATE NETWORK RULE chatgpt_network_rule
  MODE = INGRESS
  TYPE = IPV4
  VALUE_LIST = ('23.102.140.112/28',...,'40.84.180.128/28');

-- Create a network policy that uses this rule
CREATE NETWORK POLICY chatgpt_network_policy
  ALLOWED_NETWORK_RULE_LIST = ('chatgpt_network_rule');
```

> **Note:** Network policies can be applied at multiple levels (account, user). If you encounter error `390422` or "Invalid Client", you may need to adjust policies for your specific user or security integration.

### Step 2.2: Create the OAuth Security Integration
Create a security integration in Snowflake to handle OAuth authentication for your GPT Action. Use a temporary redirect URI for initial testing.

```sql
CREATE SECURITY INTEGRATION CHATGPT_INTEGRATION
  TYPE = OAUTH
  ENABLED = TRUE
  OAUTH_CLIENT = CUSTOM
  OAUTH_CLIENT_TYPE = 'CONFIDENTIAL'
  OAUTH_REDIRECT_URI = 'https://oauth.pstmn.io/v1/callback' -- Temporary test URI
  OAUTH_ISSUE_REFRESH_TOKENS = TRUE
  OAUTH_REFRESH_TOKEN_VALIDITY = 7776000
  NETWORK_POLICY = chatgpt_network_policy; -- Include only if you created a network policy
```

### (Optional) Step 2.3: Automate IP List Updates
ChatGPT's IP ranges update irregularly. You can automate keeping your network rule current with the following Snowflake objects.

1.  **Create a network rule for outbound access** (to fetch the IP list):
    ```sql
    CREATE OR REPLACE NETWORK RULE chatgpt_actions_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('openai.com:443');
    ```

2.  **Create an external access integration:**
    ```sql
    CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION chatgpt_actions_integration
    ALLOWED_NETWORK_RULES = (chatgpt_actions_rule)
    ENABLED = TRUE;
    ```

3.  **Create a Python UDF to fetch the latest IP ranges:**
    ```sql
    CREATE OR REPLACE FUNCTION getChatGPTActionsAddresses()
    RETURNS ARRAY
    LANGUAGE PYTHON
    RUNTIME_VERSION = 3.10
    PACKAGES = ('requests')
    EXTERNAL_ACCESS_INTEGRATIONS = (chatgpt_actions_integration)
    HANDLER = 'get_ip_address_ranges'
    AS
    $$
    import requests
    def get_ip_address_ranges():
        resp = requests.get("https://openai.com/chatgpt-actions.json", timeout=10)
        resp.raise_for_status()
        data = [entry["ipv4Prefix"] for entry in resp.json().get("prefixes", []) if "ipv4Prefix" in entry]
        return data
    $$;
    ```

4.  **Create a procedure to update the network rule:**
    ```sql
    CREATE OR REPLACE PROCEDURE update_chatgpt_network_rule()
    RETURNS STRING
    LANGUAGE SQL
    AS
    $$
    DECLARE
      ip_list STRING;
    BEGIN
      ip_list := '''' || ARRAY_TO_STRING(getChatGPTActionsAddresses(), ''',''') || '''';
      EXECUTE IMMEDIATE
        'ALTER NETWORK RULE chatgpt_network_rule SET VALUE_LIST = (' || ip_list || ')';
      RETURN 'chatgpt_network_rule updated with ' || ARRAY_SIZE(getChatGPTActionsAddresses()) || ' entries';
    END;
    $$;
    ```

5.  **Test the update procedure:**
    ```sql
    CALL update_chatgpt_network_rule();
    ```

6.  **(Optional) Schedule a daily task:**
    ```sql
    CREATE OR REPLACE TASK auto_update_chatgpt_network_rule
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 6 * * * America/Los_Angeles'
    AS
      CALL update_chatgpt_network_rule();
    ```

---

## Part 3: Configure GPT Action Authentication

### Step 3.1: Gather OAuth Credentials from Snowflake
Run the following commands in Snowflake to retrieve the necessary OAuth parameters.

1.  **Describe the integration** to get the Auth and Token URLs:
    ```sql
    DESCRIBE SECURITY INTEGRATION CHATGPT_INTEGRATION;
    ```
    Note the values for `OAUTH_AUTHORIZATION_ENDPOINT` and `OAUTH_TOKEN_ENDPOINT`.

2.  **Retrieve the Client ID and Secret:**
    ```sql
    SELECT
    trim(parse_json(SYSTEM$SHOW_OAUTH_CLIENT_SECRETS('CHATGPT_INTEGRATION')):OAUTH_CLIENT_ID) AS OAUTH_CLIENT_ID,
    trim(parse_json(SYSTEM$SHOW_OAUTH_CLIENT_SECRETS('CHATGPT_INTEGRATION')):OAUTH_CLIENT_SECRET) AS OAUTH_CLIENT_SECRET;
    ```

> **Tip:** Test your OAuth setup using a tool like Postman before proceeding. Ensure your machine's IP is allowed if you applied a network policy.

### Step 3.2: Set OAuth Values in the GPT Action
In your Custom GPT's **Authentication** settings:
1.  Select **OAuth** as the Authentication Type.
2.  Fill in the form with the values you gathered:

| Form Field | Value to Enter |
| :--- | :--- |
| Client ID | `OAUTH_CLIENT_ID` from `SHOW_OAUTH_CLIENT_SECRETS` |
| Client Secret | `OAUTH_CLIENT_SECRET` from `SHOW_OAUTH_CLIENT_SECRETS` |
| Authorization URL | `OAUTH_AUTHORIZATION_ENDPOINT` from `DESCRIBE SECURITY INTEGRATION` |
| Token URL | `OAUTH_TOKEN_ENDPOINT` from `DESCRIBE SECURITY INTEGRATION` |
| Scope | `session:role:CHATGPT_INTEGRATION_ROLE` (Replace with your role) |
| Token Exchange Method | Default (POST Request) |

**Scope Note:** The format is `session:role:<your_role_name>`. Specifying it here includes it in the OAuth consent screen, which can improve reliability.

---

## Part 4: Finalize the Integration

### Step 4.1: Update the Snowflake Redirect URI
After setting up authentication in ChatGPT, it will provide a final callback URL.

1.  Copy the **callback URL** from the GPT Action's authentication section.
2.  Update your Snowflake security integration to use this final URL:

```sql
ALTER SECURITY INTEGRATION CHATGPT_INTEGRATION
SET OAUTH_REDIRECT_URI='https://chat.openai.com/aip/<callback_id>/oauth/callback';
```

Replace the entire URL string with the one provided by ChatGPT.

---

## FAQ & Troubleshooting

*   **General Support:** This guide illustrates integration concepts. Full support for third-party API configurations is not provided.
*   **Callback URL Error:** If authentication fails with a callback error, double-check Step 4.1. The exact URL from ChatGPT must be set in your `CHATGPT_INTEGRATION`.
*   **Incorrect Warehouse/Database:** If the GPT calls the wrong resources, refine your **Instructions** (Step 1.1) to be more explicit about default values or to require the user to specify them.
*   **Changing the OpenAPI Schema:** If you update your GPT's Action YAML, the callback URL may change. Re-verify and update the Redirect URI in Snowflake if needed.

Your GPT Action for Snowflake is now configured. You can start a conversation with your Custom GPT, and it will use the defined workflow to query your data warehouse.