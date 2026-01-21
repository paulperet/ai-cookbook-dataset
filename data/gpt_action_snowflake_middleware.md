# Building a GPT Action for Snowflake Data Analysis

## Introduction

This guide walks you through building a GPT Action that connects ChatGPT to a Snowflake Data Warehouse. The action enables ChatGPT to execute SQL queries and retrieve results as CSV files for use with Data Analysis (Code Interpreter). You'll implement a middleware Azure Function that formats Snowflake responses for ChatGPT.

### Prerequisites

Before starting, ensure you have:
- A provisioned Snowflake Data Warehouse
- User access to necessary databases, schemas, and tables
- Azure Portal or VS Code access for creating Azure Function Apps
- Familiarity with [GPT Actions](https://platform.openai.com/docs/actions)

## Step 1: Configure OAuth in Azure Entra ID

First, set up OAuth authentication between Azure and Snowflake.

### 1.1 Create App Registration

1. Navigate to the [Microsoft Azure Portal](https://portal.azure.com/)
2. Go to **Azure Entra ID** → **App Registrations** → **New Registration**
3. Enter `Snowflake GPT OAuth Client` as the name
4. Set **Supported account types** to **Single Tenant**
5. Click **Register**

### 1.2 Collect Configuration Values

After registration, note these values from the **Overview** section:

```python
# Essential OAuth configuration values
TENANT_ID = "your-tenant-id-here"  # Directory (tenant) ID
CLIENT_ID = "your-client-id-here"  # Application (client) ID

# Derived values
AZURE_AD_ISSUER = f"https://sts.windows.net/{TENANT_ID}/"
AZURE_AD_JWS_KEY_ENDPOINT = f"https://login.microsoftonline.com/{TENANT_ID}/discovery/v2.0/keys"
```

From the **Endpoints** page, note:
- **OAuth 2.0 authorization endpoint (v2)** as `AZURE_AD_OAUTH_AUTHORIZATION_ENDPOINT`
- **OAuth 2.0 token endpoint (v2)** as `AZURE_AD_OAUTH_TOKEN_ENDPOINT`

### 1.3 Configure API Scopes

1. Go to **Expose an API**
2. Set the **Application ID URI** (e.g., `https://your.company.com/unique-id`)
3. Click **Add a scope** with these settings:
   - Scope name: `session:scope:analyst` (replace `analyst` with your Snowflake role)
   - Display name: `Account Admin`
   - Description: `Can administer the Snowflake account`
4. Save the scope as `AZURE_AD_SCOPE` (concatenation of Application ID URI and scope name)

### 1.4 Create Client Secret

1. Go to **Certificates & secrets** → **New client secret**
2. Add description and select **730 days (24 months)**
3. Copy the secret value as `OAUTH_CLIENT_SECRET`

### 1.5 Configure API Permissions

1. Go to **API Permissions** → **Add Permission** → **My APIs**
2. Select your **Snowflake OAuth Resource**
3. Check **Delegated Permissions** for the scope you created
4. Click **Add Permissions**
5. Click **Grant Admin Consent** → **Yes**

## Step 2: Create Security Integration in Snowflake

Link your Azure App Registration to Snowflake using an External OAuth Security Integration.

```sql
CREATE OR REPLACE SECURITY INTEGRATION AZURE_OAUTH_INTEGRATION
  TYPE = EXTERNAL_OAUTH
  ENABLED = TRUE
  EXTERNAL_OAUTH_TYPE = 'AZURE'
  EXTERNAL_OAUTH_ISSUER = '<AZURE_AD_ISSUER>'
  EXTERNAL_OAUTH_JWS_KEYS_URL = '<AZURE_AD_JWS_KEY_ENDPOINT>'
  EXTERNAL_OAUTH_AUDIENCE_LIST = ('<SNOWFLAKE_APPLICATION_ID_URI>')
  EXTERNAL_OAUTH_TOKEN_USER_MAPPING_CLAIM = 'upn'
  EXTERNAL_OAUTH_SNOWFLAKE_USER_MAPPING_ATTRIBUTE = 'EMAIL_ADDRESS';
```

Replace the placeholders with values from Step 1.

## Step 3: Build the Azure Function App

Create an Azure Function that executes SQL queries and returns CSV files to ChatGPT.

### 3.1 Setup and Imports

First, install required packages:

```bash
pip install azure-functions azure-storage-blob snowflake-connector-python pyjwt
```

Then import them in your function:

```python
import azure.functions as func
import logging
import tempfile
import csv
import datetime
import jwt
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
import snowflake.connector
```

### 3.2 Connect to Snowflake

Extract the access token and establish a Snowflake connection:

```python
def connect_to_snowflake(req):
    """Extract token and connect to Snowflake."""
    # Extract the token from the Authorization header
    auth_header = req.headers.get('Authorization')
    token_type, token = auth_header.split()

    try:
        # Extract email address from token for Snowflake user mapping
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        email = decoded_token.get('upn')
        
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=email,  # Snowflake username (email in this example)
            account=os.environ['SNOWFLAKE_ACCOUNT'],  # Your Snowflake account
            authenticator="oauth",
            token=token,
            warehouse=os.environ['SNOWFLAKE_WAREHOUSE']  # Your warehouse
        )
        logging.info("Successfully connected to Snowflake.")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to Snowflake: {e}")
        raise
```

### 3.3 Execute Query and Create CSV

Execute the SQL query and convert results to CSV format:

```python
def execute_query_and_create_csv(conn, sql_query):
    """Execute SQL query and save results as CSV."""
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        logging.info(f"Query executed successfully: {sql_query}")

        # Convert results to CSV
        csv_file_path = write_results_to_csv(results, column_names)
        return csv_file_path
    except Exception as e:
        logging.error(f"Error executing query or processing data: {e}")
        raise

def write_results_to_csv(results, column_names):
    """Write query results to a temporary CSV file."""
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
        csv_writer = csv.writer(temp_file)
        csv_writer.writerow(column_names)  # Write column headers
        csv_writer.writerows(results)      # Write data rows
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logging.error(f"Error writing results to CSV: {e}")
        raise
```

### 3.4 Upload to Azure Blob Storage

Upload the CSV to Azure Blob Storage and generate a secure URL:

```python
def upload_csv_to_azure(file_path, container_name, blob_name, connect_str):
    """Upload CSV to Azure Blob Storage and generate SAS URL."""
    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Upload file with proper content settings
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(
                content_type='text/csv',
                content_disposition=f'attachment; filename="{blob_name}"'
            ))
        logging.info(f"Successfully uploaded {file_path} to {container_name}/{blob_name}")

        # Generate SAS token (valid for 1 hour)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )

        # Return the full URL with SAS token
        return f"{blob_client.url}?{sas_token}"
    except Exception as e:
        logging.error(f"Error uploading to Azure Blob Storage: {e}")
        raise
```

### 3.5 Main Function Handler

Combine all components into the main Azure Function:

```python
def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main Azure Function handler."""
    try:
        # Extract SQL query from request
        sql_query = req.params.get('sql_query')
        if not sql_query:
            return func.HttpResponse("SQL query parameter is required", status_code=400)

        # Connect to Snowflake
        conn = connect_to_snowflake(req)
        
        # Execute query and create CSV
        csv_file_path = execute_query_and_create_csv(conn, sql_query)
        
        # Upload to Azure Blob Storage
        connect_str = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        container_name = os.environ['AZURE_STORAGE_CONTAINER']
        blob_name = f"query_result_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        file_url = upload_csv_to_azure(csv_file_path, container_name, blob_name, connect_str)
        
        # Clean up temporary file
        os.unlink(csv_file_path)
        
        # Return response in OpenAI file response format
        response_body = {
            "openaiFileResponse": [
                {
                    "file_name": blob_name,
                    "file_url": file_url,
                    "mime_type": "text/csv"
                }
            ]
        }
        
        return func.HttpResponse(
            json.dumps(response_body),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        return func.HttpResponse(f"Error processing request: {str(e)}", status_code=500)
```

## Step 4: Configure Environment Variables

Set these environment variables in your Azure Function:

```bash
SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
SNOWFLAKE_WAREHOUSE=your_warehouse
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_CONTAINER=your-container-name
```

## Step 5: Test Your Integration

1. Deploy your Azure Function
2. Test with a sample SQL query
3. Verify the CSV file is generated and accessible via URL
4. Check that the response follows the OpenAI file response format

## Conclusion

You've now built a complete GPT Action that:
1. Authenticates users via Azure Entra ID OAuth
2. Executes SQL queries in Snowflake
3. Returns results as downloadable CSV files
4. Integrates with ChatGPT's Data Analysis feature

This enables ChatGPT to query your Snowflake data warehouse and perform advanced analysis on the results, all while maintaining proper authentication and security controls.

For production deployment, consider adding:
- Query sanitization and validation
- Rate limiting
- Enhanced error handling
- Monitoring and logging
- Caching mechanisms for frequently accessed data