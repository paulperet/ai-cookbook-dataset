# Guide: Build a GPT Action to Query AWS Redshift

## Introduction

This guide walks you through creating a GPT Action that connects to an AWS Redshift data warehouse. The solution uses an AWS Lambda function as middleware to execute SQL queries securely within your VPC and return the results as downloadable files to ChatGPT. This enables natural language interaction with your analytical data.

**Prerequisites:**
- Access to an AWS Redshift environment (Serverless or provisioned)
- Permissions to deploy AWS resources (Lambda, API Gateway, VPC, Cognito)
- AWS CLI authenticated on your local machine
- Basic familiarity with AWS Serverless Application Model (SAM)

## Part 1: Middleware Setup

### Step 1: Install Required Tools

Ensure you have the following installed locally:

```bash
# AWS Command Line Interface
# Follow: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

# AWS SAM CLI
# Follow: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html

# Python 3.11 or later

# yq (YAML processor)
# Installation: https://github.com/mikefarah/yq?tab=readme-ov-file#install
```

### Step 2: Create the Lambda Function Code

Create a new directory for your project and add the following Python code as `app.py`:

```python
import json
import psycopg2
import os
import base64
import tempfile
import csv

# Fetch Redshift credentials from environment variables
host = os.environ['REDSHIFT_HOST']
port = os.environ['REDSHIFT_PORT']
user = os.environ['REDSHIFT_USER']
password = os.environ['REDSHIFT_PASSWORD']
database = os.environ['REDSHIFT_DB']

def execute_statement(sql_statement):
    try:
        # Establish connection
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=database
        )
        cur = conn.cursor()
        cur.execute(sql_statement)
        conn.commit()

        # Fetch all results
        if cur.description:
            columns = [desc[0] for desc in cur.description]
            result = cur.fetchall()
        else:
            columns = []
            result = []

        cur.close()
        conn.close()
        return columns, result

    except Exception as e:
        raise Exception(f"Database query failed: {str(e)}")

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])
        sql_statement = data['sql_statement']

        # Execute the statement and fetch results
        columns, result = execute_statement(sql_statement)
        
        # Create a temporary file to save the result as CSV
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv', newline='') as tmp_file:
            csv_writer = csv.writer(tmp_file)
            if columns:
                csv_writer.writerow(columns)  # Write the header
            csv_writer.writerows(result)  # Write all rows
            tmp_file_path = tmp_file.name

        # Read the file and encode its content to base64
        with open(tmp_file_path, 'rb') as f:
            file_content = f.read()
            encoded_content = base64.b64encode(file_content).decode('utf-8')

        response = {
            'openaiFileResponse': [
                {
                    'name': 'query_result.csv',
                    'mime_type': 'text/csv',
                    'content': encoded_content
                }
            ]
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**Note:** This function uses the `psycopg2` library. You'll need to include it in your deployment package, typically via a `requirements.txt` file.

### Step 3: Retrieve Redshift Network Configuration

Your Lambda function must be deployed in the same VPC as your Redshift cluster. Retrieve the necessary network details using the AWS CLI:

```bash
# For Redshift Serverless
aws redshift-serverless get-workgroup \
  --workgroup-name default-workgroup \
  --query 'workgroup.{address: endpoint.address, port: endpoint.port, SecurityGroupIds: securityGroupIds, SubnetIds: subnetIds}'
```

Save the output. You'll need:
- The endpoint address (host)
- The port (typically 5439)
- One Security Group ID
- At least two Subnet IDs (the example uses six for high availability)

### Step 4: Configure Environment Variables

Create a file named `env.yaml` with the following structure, replacing the placeholder values with your actual Redshift credentials and network information:

```yaml
RedshiftHost: default-workgroup.xxxxx.{region}.redshift-serverless.amazonaws.com
RedshiftPort: 5439
RedshiftUser: username
RedshiftPassword: password
RedshiftDb: my-db
SecurityGroupId: sg-xx
SubnetId1: subnet-xx
SubnetId2: subnet-xx
SubnetId3: subnet-xx
SubnetId4: subnet-xx
SubnetId5: subnet-xx
SubnetId6: subnet-xx
```

### Step 5: Create the SAM Template

Create a `template.yaml` file with the following CloudFormation/SAM configuration:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: >
  redshift-middleware

  Middleware to fetch RedShift data and return it through HTTP as files

Globals:
  Function:
    Timeout: 3

Parameters:
  RedshiftHost:
    Type: String
  RedshiftPort:
    Type: String
  RedshiftUser:
    Type: String
  RedshiftPassword:
    Type: String
  RedshiftDb:
    Type: String
  SecurityGroupId:
    Type: String
  SubnetId1:
    Type: String
  SubnetId2:
    Type: String
  SubnetId3:
    Type: String
  SubnetId4:
    Type: String
  SubnetId5:
    Type: String
  SubnetId6:
    Type: String
  CognitoUserPoolName:
    Type: String
    Default: MyCognitoUserPool
  CognitoUserPoolClientName:
    Type: String
    Default: MyCognitoUserPoolClient

Resources:
  MyCognitoUserPool:
    Type: AWS::Cognito::UserPool
    Properties:
      UserPoolName: !Ref CognitoUserPoolName
      Policies:
        PasswordPolicy:
          MinimumLength: 8
      UsernameAttributes:
        - email
      Schema:
        - AttributeDataType: String
          Name: email
          Required: false

  MyCognitoUserPoolClient:
    Type: AWS::Cognito::UserPoolClient
    Properties:
      UserPoolId: !Ref MyCognitoUserPool
      ClientName: !Ref CognitoUserPoolClientName
      GenerateSecret: true

  RedshiftMiddlewareApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: Prod
      Cors: "'*'"
      Auth:
        DefaultAuthorizer: MyCognitoAuthorizer
        Authorizers:
          MyCognitoAuthorizer:
            AuthorizationScopes:
              - openid
              - email
              - profile
            UserPoolArn: !GetAtt MyCognitoUserPool.Arn
        
  RedshiftMiddlewareFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./
      Handler: app.lambda_handler
      Runtime: python3.11
      Timeout: 45
      Architectures:
        - x86_64
      Events:
        SqlStatement:
          Type: Api
          Properties:
            Path: /sql_statement
            Method: post
            RestApiId: !Ref RedshiftMiddlewareApi
      Environment:
        Variables:
          REDSHIFT_HOST: !Ref RedshiftHost
          REDSHIFT_PORT: !Ref RedshiftPort
          REDSHIFT_USER: !Ref RedshiftUser
          REDSHIFT_PASSWORD: !Ref RedshiftPassword
          REDSHIFT_DB: !Ref RedshiftDb
      VpcConfig:
        SecurityGroupIds:
          - !Ref SecurityGroupId
        SubnetIds:
          - !Ref SubnetId1
          - !Ref SubnetId2
          - !Ref SubnetId3
          - !Ref SubnetId4
          - !Ref SubnetId5
          - !Ref SubnetId6

Outputs:
  RedshiftMiddlewareApi:
    Description: "API Gateway endpoint URL for Prod stage for SQL Statement function"
    Value: !Sub "https://${RedshiftMiddlewareApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/sql_statement/"
  RedshiftMiddlewareFunction:
    Description: "SQL Statement Lambda Function ARN"
    Value: !GetAtt RedshiftMiddlewareFunction.Arn
  RedshiftMiddlewareFunctionIamRole:
    Description: "Implicit IAM Role created for SQL Statement function"
    Value: !GetAtt RedshiftMiddlewareFunctionRole.Arn
  CognitoUserPoolArn:
    Description: "ARN of the Cognito User Pool"
    Value: !GetAtt MyCognitoUserPool.Arn
```

### Step 6: Deploy the Application

Use the AWS SAM CLI to deploy your stack. First, prepare the parameters:

```bash
PARAM_FILE="env.yaml"
PARAMS=$(yq eval -o=json $PARAM_FILE | jq -r 'to_entries | map("\(.key)=\(.value|tostring)") | join(" ")')
```

Then deploy:

```bash
sam deploy \
  --template-file template.yaml \
  --stack-name redshift-middleware \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides $PARAMS
```

After deployment, note the API endpoint URL from the output. You'll need this for the GPT Action configuration.

### Step 7: Test the Endpoint

Verify your deployment by making a test request:

```bash
curl -X POST https://<your-api-url>/Prod/sql_statement/ \
  -H "Content-Type: application/json" \
  -d '{ "sql_statement": "SELECT * FROM customers LIMIT 10" }'
```

You should receive a JSON response containing a base64-encoded CSV file.

## Part 2: Configure the GPT Action

### Step 8: Create a Custom GPT

In the ChatGPT interface, create a new Custom GPT. You'll configure two main sections: Instructions and Actions.

### Step 9: Add System Instructions

In the **Instructions** panel, paste the following:

```
**Context**: You are an expert at writing Redshift SQL queries. You will initially retrieve the table schema that you will use thoroughly. Every attribute, table name, or data type will be known by you.

**Instructions**:
1. No matter the user's question, start by running `runQuery` operation using this query: "SELECT table_name, column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;" It will help you understand how to query the data. A CSV will be returned with all the attributes and their table. Make sure to read it fully and understand all available tables & their attributes before querying. You don't have to show this to the user.
2. Convert the user's question into a SQL statement that leverages the step above and run the `runQuery` operation on that SQL statement to confirm the query works. Let the user know which table you will use/query.
3. Execute the query and show them the data. Show only the first few rows.

**Additional Notes**: If the user says "Let's get started", explain they can ask a question they want answered about data that we have access to. If the user has no ideas, suggest that we have transactions data they can query - ask if they want you to query that.
**Important**: Never make up a table name or table attribute. If you don't know, go back to the data you've retrieved to check what is available. If you think no table or attribute is available, then tell the user you can't perform this query for them.
```

### Step 10: Configure the Action Schema

In the **Actions** panel, add a new action and paste the following OpenAPI schema. Replace `{your_function_url}` with your actual API Gateway endpoint (without the trailing `/sql_statement`).

```yaml
openapi: 3.1.0
info:
  title: SQL Execution API
  description: API to execute SQL statements and return results as a file.
  version: 1.0.0
servers:
  - url: {your_function_url}/Prod
    description: Production server
paths:
  /sql_statement:
    post:
      operationId: executeSqlStatement
      summary: Executes a SQL statement and returns the result as a file.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                sql_statement:
                  type: string
                  description: The SQL statement to execute.
                  example: SELECT * FROM customers LIMIT 10
              required:
                - sql_statement
      responses:
        '200':
          description: The SQL query result as a JSON file.
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
                          example: query_result.json
                        mime_type:
                          type: string
                          description: The MIME type of the file.
                          example: application/json
                        content:
                          type: string
                          description: The base64 encoded content of the file.
                          format: byte
                          example: eyJrZXkiOiJ2YWx1ZSJ9
        '500':
          description: Error response
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
                    description: Error message.
                    example: Database query failed error details
```

### Step 11: Configure Authentication

Follow the authentication setup instructions in the [AWS Middleware Cookbook](https://cookbook.openai.com/examples/chatgpt/gpt_actions_library/gpt_middleware_aws_function) to configure Cognito authentication for your GPT Action. This ensures only authorized users can execute queries.

## Conclusion

You have now deployed a complete system that allows ChatGPT to query your AWS Redshift database securely. The middleware Lambda function executes SQL within your VPC, while the Custom GPT provides a natural language interface for data exploration.

**Next Steps:**
- Add users to your Cognito User Pool for access control
- Consider adding query validation or logging in the Lambda function
- Test various natural language queries with your new GPT

This solution enables data scientists, analysts, and other users to interact with your data warehouse using conversational language, lowering the barrier to data access and exploration.