# Guide: Building a GPT Action with AWS Lambda Middleware

## Introduction

This guide walks you through creating a secure middleware function using AWS Lambda that can be integrated as a GPT Action. You'll learn to deploy a serverless function protected by OAuth authentication, enabling ChatGPT to securely interact with your AWS resources or custom logic.

### Value & Use Cases

**Core Value**: Connect ChatGPT directly to AWS services or custom backend logic through a secure, authenticated API endpoint.

**Example Use Cases**:
- Query data from AWS services like Redshift, DynamoDB, or S3 through ChatGPT.
- Pre-process API responses (e.g., overcome context limits, add metadata).
- Return files (CSV, PDF) for ChatGPT to analyze as uploaded documents.
- Trigger multi-step AWS workflows directly from a conversation.

## Prerequisites

Before starting, ensure you have:
- An AWS account with permissions to create:
  - Lambda functions
  - S3 buckets
  - CloudFormation stacks
  - API Gateway
  - Cognito User Pools
  - IAM roles
- AWS CLI installed and configured
- AWS SAM CLI installed

## Step 1: Clone the Example Repository

Begin by cloning the example AWS Lambda middleware repository:

```bash
git clone https://github.com/pap-openai/aws-lambda-middleware
cd lambda-middleware
```

## Step 2: Review the SAM Template

The Serverless Application Model (SAM) template defines your AWS infrastructure. Here's the key structure:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: AWS middleware function

Parameters:
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

  MyCognitoUserPoolClient:
    Type: AWS::Cognito::UserPoolClient
    Properties:
      UserPoolId: !Ref MyCognitoUserPool
      ClientName: !Ref CognitoUserPoolClientName
      GenerateSecret: true

  MiddlewareApi:
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
        
  MiddlewareFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: aws-middleware/
      Handler: app.lambda_handler
      Runtime: python3.11
      Timeout: 45
      Events:
        SqlStatement:
          Type: Api
          Properties:
            Path: /my_route
            Method: post
            RestApiId: !Ref MiddlewareApi
```

This template creates:
- A Cognito User Pool for authentication
- An API Gateway with Cognito authorizer
- A Lambda function with a POST endpoint at `/my_route`

## Step 3: Deploy the SAM Application

Build and deploy your serverless application:

```bash
sam build
sam deploy --template-file template.yaml --stack-name aws-middleware --capabilities CAPABILITY_IAM
```

After deployment, note the API endpoint URL from the CloudFormation outputs. You can verify the authentication is working by testing without credentials:

```bash
curl -d {} <your-api-endpoint-url>
```

This should return `{"message":"Unauthorized"}`.

## Step 4: Set Up Authentication in AWS Cognito

### Create a Test User

Retrieve your User Pool ID from the deployment outputs (format: `your-region_xxxxx`) and create a test user:

```bash
aws cognito-idp admin-create-user \
    --user-pool-id "your-region_xxxxx" \
    --username johndoe@example.com \
    --user-attributes Name=email,Value=johndoe@example.com \
    --temporary-password "TempPassword123"
```

### Configure Cognito Domain and App Client

1. Navigate to AWS Cognito in the AWS Console
2. Select your User Pool
3. Go to **App Integration** > **Domains** and create a Cognito domain
4. Under **App client list**, select your app client
5. Edit the Hosted UI configuration:
   - Add a callback URL (use a placeholder for now)
   - Set Authorization Scheme to "Authorization code grant"
   - Add OAuth scopes: `openid`, `email`, `profile`

**Note**: You'll update the callback URL later when ChatGPT provides one during action configuration.

## Step 5: Create the GPT Action

### Define the OpenAPI Specification

In ChatGPT's Action configuration, use this OpenAPI spec:

```yaml
openapi: 3.1.0
info:
  title: Success API
  description: API that returns a success message.
  version: 1.0.0
servers:
  - url: https://<your-api-endpoint>/Prod
    description: Main production server
paths:
  /my_route:
    post:
      operationId: postSuccess
      summary: Returns a success message.
      description: Endpoint to check the success status.
      responses:
        '200':
          description: A JSON object indicating success.
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                    example: true
```

Replace `<your-api-endpoint>` with your actual API Gateway URL.

### Configure OAuth Authentication

1. In the Action configuration, go to **Authentication** > **OAuth**
2. Retrieve these values from your Cognito User Pool App Client:
   - Client ID
   - Client Secret
3. Set the Token URLs:
   - Token URL: `https://<your-cognito-domain>/oauth2/token`
   - Authorization URL: `https://<your-cognito-domain>/oauth2/authorize`
4. Add `openid` to the scope
5. Save the configuration

## Step 6: Connect Cognito with ChatGPT

1. After saving the Action, ChatGPT will display a callback URL
2. Copy this callback URL
3. Return to your Cognito User Pool App Client configuration
4. Update the Hosted UI callback URLs to include ChatGPT's URL
5. Save the changes

## Step 7: Test the Integration

Now test your GPT Action:

1. In ChatGPT, trigger your Action
2. You'll be redirected to AWS Cognito's login page
3. Log in with your test user credentials
4. ChatGPT will successfully call your Lambda function and return the response

## Conclusion

You've successfully created a secure middleware function that connects ChatGPT to AWS services through OAuth-protected API endpoints. This foundation allows you to:

1. Extend the Lambda function to interact with other AWS services
2. Customize the authentication flow (e.g., connect to your existing identity provider)
3. Modify the SAM template to fit your specific infrastructure needs

For a more complex example, refer to the [RedShift cookbook](https://cookbook.openai.com/examples/chatgpt/gpt_actions_library/gpt_middleware_aws_function) which follows similar patterns but includes database connectivity.

**Best Practice**: For production use, consider integrating Cognito with your existing identity provider rather than managing users directly in Cognito.