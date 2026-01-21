# Building a GPT Action for Salesforce Service Cloud: A Developer's Guide

## Introduction

This guide walks you through creating a custom GPT Action that connects ChatGPT directly to Salesforce Service Cloud. This integration enables users to query case information and update case status using natural language, streamlining customer support workflows.

### Key Benefits
- **Reduce Response Time**: Quickly access and update case information without navigating Salesforce UI
- **Improve Consistency**: Maintain consistent brand voice with GPT-powered responses
- **Streamline Workflows**: Handle common support tasks directly through conversation

### Prerequisites
Before starting, ensure you have:
- Access to a Salesforce instance with Service Cloud
- Permissions to create Connected Apps in Salesforce
- Familiarity with [GPT Actions basics](https://platform.openai.com/docs/actions/introduction)

## Step 1: Configure Your Custom GPT Instructions

When creating your Custom GPT, add the following instructions to guide its behavior:

```markdown
**Context**: Your purpose is to pull information from Service Cloud, and push updates to cases. A user is going to ask you a question and ask you to make updates.

**Instructions**:
1. When a user asks you to help them solve a case in Service Cloud, ask for the case number and pull the details for the case into the conversation using the getCaseDetailsFromNumber action.
2. If the user asks you to update the case details, use the action updateCaseStatus.

**Example**: 
User: Help me solve case 00001104 in Service Cloud.
```

These instructions ensure your GPT understands its role and knows when to invoke the Salesforce API actions.

## Step 2: Define the OpenAPI Schema

In the Actions panel of your Custom GPT, paste the following OpenAPI schema. This defines the API endpoints your GPT will use to interact with Salesforce.

```yaml
openapi: 3.1.0
info:
  title: Salesforce Service Cloud Case Update API
  description: API for updating the status of Service Cloud tickets (cases) in Salesforce.
  version: 1.0.3
servers:
  - url: https://your_instance.my.salesforce.com
    description: Base URL for your Salesforce instance (replace 'your_instance' with your actual Salesforce domain)
paths:
  /services/data/v60.0/sobjects/Case/{CaseId}:
    patch:
      operationId: updateCaseStatus
      summary: Updates the status of a Service Cloud case
      description: Updates the status of a Service Cloud ticket based on the case ID number.
      parameters:
        - name: CaseId
          in: path
          required: true
          description: The ID of the case to update.
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                Status:
                  type: string
                  description: The new status of the case.
      responses:
        '204':
          description: Successfully updated the case status
        '400':
          description: Bad request - invalid input or case ID not found
        '401':
          description: Unauthorized - authentication required
        '404':
          description: Not Found - case ID does not exist
    delete:
      operationId: deleteCase
      summary: Deletes a Service Cloud case
      description: Deletes a Service Cloud ticket based on the case ID number.
      parameters:
        - name: CaseId
          in: path
          required: true
          description: The ID of the case to delete.
          schema:
            type: string
      responses:
        '204':
          description: Successfully deleted the case
        '400':
          description: Bad request - invalid case ID
        '401':
          description: Unauthorized - authentication required
        '404':
          description: Not Found - case ID does not exist
  /services/data/v60.0/query:
    get:
      operationId: getCaseDetailsFromNumber
      summary: Retrieves case details using a case number
      description: Retrieves the details of a Service Cloud case associated with a given case number.
      parameters:
        - name: q
          in: query
          required: true
          description: SOQL query string to find the Case details based on Case Number.
          schema:
            type: string
            example: "SELECT Id, CaseNumber, Status, Subject, Description FROM Case WHERE CaseNumber = '123456'"
      responses:
        '200':
          description: Successfully retrieved the case details
          content:
            application/json:
              schema:
                type: object
                properties:
                  totalSize:
                    type: integer
                  done:
                    type: boolean
                  records:
                    type: array
                    items:
                      type: object
                      properties:
                        Id:
                          type: string
                        CaseNumber:
                          type: string
                        Status:
                          type: string
                        Subject:
                          type: string
                        Description:
                          type: string
        '400':
          description: Bad request - invalid query
        '401':
          description: Unauthorized - authentication required
        '404':
          description: Not Found - case number does not exist
```

**Important**: Replace `your_instance` in the server URL with your actual Salesforce domain.

## Step 3: Configure OAuth Authentication in Salesforce

Before setting up authentication in ChatGPT, you need to create a Connected App in Salesforce.

### 3.1 Create a Connected App in Salesforce

1. Navigate to **Salesforce Setup**
2. Search for and select **"App Manager"**
3. Click **"New Connected App"**
4. Configure the following settings:
   - **Connected App Name**: Choose a descriptive name
   - **Contact Email**: Enter your email address
   - **Enable OAuth Settings**: Check this box
   - **Callback URL**: Use a placeholder like `https://chat.openai.com/aip//oauth/callback` (you'll update this later)
   - **Selected OAuth Scopes**: Grant appropriate permissions based on your security policies

5. Ensure the following OAuth settings are configured:
   - ✅ Enable Client Credentials Flow
   - ✅ Enable Authorization Code and Credentials Flow
   - ✅ Enable Token Exchange Flow
   - ❌ **Uncheck**: Require Proof Key for Code Exchange (PKCE) Extension for Supported Authorization Flows

6. Click **"Save"**

### 3.2 Retrieve Your Credentials

1. On your Connected App page, find **"Consumer Key and Secret"**
2. Click **"Manage Consumer Details"**
3. Verify your access using the code emailed to your account
4. Copy the following credentials:
   - **Consumer Key** = ChatGPT Client ID
   - **Consumer Secret** = ChatGPT Client Secret

### 3.3 Configure OAuth Policies

1. Return to your Connected App page
2. Click **"Manage"**
3. Click **"Edit Policies"**
4. Under OAuth Policies, check **"Enable Token Exchange Flow"**
5. Click **"Save"**

## Step 4: Configure Authentication in ChatGPT

In your Custom GPT's Authentication settings:

1. Select **"OAuth"** as the authentication type
2. Enter the following information:
   - **Client ID**: Paste the Consumer Key from Salesforce
   - **Client Secret**: Paste the Consumer Secret from Salesforce
   - **Authorization URL**: `https://[your-instance].my.salesforce.com/services/oauth2/authorize`
   - **Token URL**: `https://[your-instance].my.salesforce.com/services/oauth2/token`
   - **Scope**: `full`
   - **Token**: Default (POST)

**Note**: Replace `[your-instance]` with your actual Salesforce domain.

## Step 5: Finalize the Configuration

### 5.1 Update Callback URL in Salesforce

1. Copy the callback URL from your GPT Action in ChatGPT
2. Navigate back to your Connected App in Salesforce
3. Edit the Connected App settings
4. Replace the placeholder callback URL with the actual URL from ChatGPT
5. Save your changes

## Troubleshooting Common Issues

### Callback URL Error
If you encounter a callback URL error in ChatGPT:
- Ensure you've added the exact callback URL from ChatGPT into your Salesforce Connected App settings
- Verify there are no typos or extra characters in the URL

### Internal Server Error
If you get an internal server error:
- Double-check all OAuth settings in your Salesforce Connected App
- Ensure "Enable Token Exchange Flow" is checked
- Verify that PKCE is **not** required (this box should be unchecked)

### Authentication Failures
- Confirm your Consumer Key and Secret are correctly copied
- Verify your Salesforce instance URL is correct in both the OpenAPI schema and authentication URLs
- Ensure you have the necessary permissions in Salesforce

## Next Steps

Your GPT Action is now configured! Test it by asking your Custom GPT to:
1. Retrieve details for a specific case number
2. Update the status of an existing case

Remember that this schema is specific to Salesforce Service Cloud cases. To connect to other Salesforce Clouds, you'll need to modify the API schema while using the same Connected App and authentication setup.

## Additional Resources

- [Salesforce API Documentation](https://developer.salesforce.com/docs/apis)
- [GPT Actions Getting Started Guide](https://platform.openai.com/docs/actions/getting-started)
- [Salesforce OAuth Documentation](https://help.salesforce.com/s/articleView?id=sf.remoteaccess_oauth_tokens_scopes.htm&type=5)

Need help or have suggestions? File an issue or PR in our GitHub repository for assistance with this integration.