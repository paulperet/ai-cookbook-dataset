# Building a GPT Action for Jira: A Step-by-Step Guide

## Introduction

This guide walks you through creating a GPT Action that connects ChatGPT to Jira Cloud. This integration enables users to manage Jira issues using natural language, allowing them to create, read, update, and assign issues directly through conversation.

### Prerequisites

Before you begin, ensure you have:
- An active Jira Cloud instance with administrative access
- Access to the [Atlassian Cloud Developer Console](https://developer.atlassian.com/console/myapps/)
- A ChatGPT Plus or Enterprise subscription with GPT Builder access

## Step 1: Set Up Your Jira OAuth Application

### 1.1 Create a New OAuth 2.0 Integration

1. Navigate to the [Atlassian Developer Console](https://developer.atlassian.com/console/myapps/)
2. Click **Create** and select **OAuth 2.0 Integration**
3. Enter a name for your integration (e.g., "ChatGPT Jira Integration")
4. Click **Create**

### 1.2 Configure Application Permissions

1. In your new application, open the **Permissions** menu from the sidebar
2. Locate **Jira API** and click **Add** → **Configure**
3. Click **Edit Scopes** under **Jira platform REST API**
4. Select the following scopes:
   - `read:jira-work`
   - `write:jira-work`
   - `read:jira-user`
5. Click **Save**

### 1.3 Add a Placeholder Callback URL

1. Click **Authorization** in the sidebar
2. Click **Configure** next to **OAuth 2.0 (3LO)**
3. Enter a placeholder URL (e.g., `https://example.com/callback`)
4. Click **Save Changes**

### 1.4 Retrieve Your Client Credentials

1. Click **Settings** in the sidebar
2. Scroll down to **Authentication Details**
3. Note down the **Client ID** and **Secret** values
   - Keep this page open for the next steps

## Step 2: Configure the GPT Action in ChatGPT

### 2.1 Create a New Custom GPT

1. Navigate to [ChatGPT's GPT Builder](https://chat.openai.com/gpts/editor)
2. Click **Create a GPT**
3. Configure the basic settings for your GPT

### 2.2 Add Custom Instructions

In the **Instructions** panel, paste the following:

```
**Context**: You are a specialized GPT designed to create and edit issues through API connections to Jira Cloud. This GPT can create, read, and edit project issues based on user instructions.

**Instructions**:
- When asked to perform a task, use the available actions via the api.atlassian.com API.
- When asked to create an issue, use the user's input to synthesize a summary and description and file the issue in JIRA.
- When asked to create a subtask, assume the project key and parent issue key of the currently discussed issue. Clarify if this context is not available.
- When asked to assign an issue or task to the user, first use JQL to query the current user's profile and use this account as the assignee.
- Ask for clarification when needed to ensure accuracy and completeness in fulfilling user requests.
```

### 2.3 Configure the OpenAPI Schema

1. In the **Actions** panel, click **Create new action**
2. In the schema editor, paste the following OpenAPI specification:

```yaml
openapi: 3.1.0
info:
  title: Jira API
  description: API for interacting with Jira issues and sub-tasks.
  version: 1.0.0
servers:
  - url: https://api.atlassian.com/ex/jira/<CLOUD_ID>/rest/api/3
    description: Jira Cloud API
components:
  securitySchemes:
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://auth.atlassian.com/authorize
          tokenUrl: https://auth.atlassian.com/oauth/token
          scopes:
            read:jira-user: Read Jira user information
            read:jira-work: Read Jira work data
            write:jira-work: Write Jira work data
  schemas:
    Issue:
      type: object
      properties:
        id:
          type: string
        key:
          type: string
        fields:
          type: object
          properties:
            summary:
              type: string
            description:
              type: string
            issuetype:
              type: object
              properties:
                name:
                  type: string
paths:
  /search:
    get:
      operationId: getIssues
      summary: Retrieve a list of issues
      parameters:
        - name: jql
          in: query
          required: false
          schema:
            type: string
        - name: startAt
          in: query
          required: false
          schema:
            type: integer
        - name: maxResults
          in: query
          required: false
          schema:
            type: integer
      responses:
        '200':
          description: A list of issues
          content:
            application/json:
              schema:
                type: object
                properties:
                  issues:
                    type: array
                    items:
                      $ref: '#/components/schemas/Issue'
  /issue:
    post:
      operationId: createIssue
      summary: Create a new issue
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                fields:
                  type: object
                  properties:
                    project:
                      type: object
                      properties:
                        key:
                          type: string
                    summary:
                      type: string
                    description:
                      type: string
                    issuetype:
                      type: object
                      properties:
                        name:
                          type: string
      responses:
        '201':
          description: Issue created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Issue'
  /issue/{issueIdOrKey}:
    get:
      operationId: getIssue
      summary: Retrieve a specific issue
      parameters:
        - name: issueIdOrKey
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Issue details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Issue'
    put:
      operationId: updateIssue
      summary: Update an existing issue
      parameters:
        - name: issueIdOrKey
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                fields:
                  type: object
                  properties:
                    summary:
                      type: string
                    description:
                      type: string
                    issuetype:
                      type: object
                      properties:
                        name:
                          type: string
      responses:
        '204':
          description: Issue updated successfully
  /issue:
    post:
      operationId: createSubTask
      summary: Create a sub-task for an issue
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                fields:
                  type: object
                  properties:
                    project:
                      type: object
                      properties:
                        key:
                          type: string
                    parent:
                      type: object
                      properties:
                        key:
                          type: string
                    summary:
                      type: string
                    description:
                      type: string
                    issuetype:
                      type: object
                      properties:
                        name:
                          type: string
      responses:
        '201':
          description: Sub-task created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Issue'
security:
  - OAuth2:
      - read:jira-user
      - read:jira-work
      - write:jira-work
```

**Important**: Replace `<CLOUD_ID>` in the server URL with your Jira Cloud instance's unique ID. You can find this value by visiting `https://<YOUR_SUBDOMAIN>.atlassian.net/_edge/tenant_info`

### 2.4 Configure OAuth Authentication

1. Click **Authentication** and select **OAuth**
2. Enter the following information:
   - **Client ID**: The Client ID from Step 1.4
   - **Client Secret**: The Secret from Step 1.4
   - **Authorization URL**: `https://auth.atlassian.com/authorize`
   - **Token URL**: `https://auth.atlassian.com/oauth/token`
   - **Scope**: `read:jira-work write:jira-work read:jira-user`
   - **Token Exchange Method**: Default (POST Request)

3. Copy the **Callback URL** provided by ChatGPT

## Step 3: Finalize Jira Configuration

### 3.1 Update Callback URL in Atlassian Console

1. Return to your application in the Atlassian Developer Console
2. Navigate to **Authorization** → **OAuth 2.0 (3LO)**
3. Click **Configure**
4. Replace the placeholder callback URL with the one copied from ChatGPT
5. Click **Save Changes**

## Step 4: Test Your Integration

### 4.1 Save and Test Your GPT

1. Click **Save** in ChatGPT GPT Builder
2. Test your GPT by asking it to perform Jira operations, such as:
   - "Show me recent issues in project PROJ"
   - "Create a new bug report about login issues"
   - "Assign task PROJ-123 to me"

### 4.2 Authorize the Connection

On first use, ChatGPT will prompt you to authorize the connection to Jira. Follow the OAuth flow to grant the necessary permissions.

## Troubleshooting

### Callback URL Errors

If you encounter callback URL errors:
1. Double-check the Callback URL value in both ChatGPT and the Atlassian Developer Console
2. Ensure they match exactly
3. Note that the callback URL may change if you modify authentication settings in ChatGPT

### Permission Issues

If actions fail due to insufficient permissions:
1. Verify all three scopes are enabled in your Atlassian application
2. Ensure the user authorizing has appropriate Jira permissions
3. Check that the project keys and issue keys referenced exist

## Next Steps

Your GPT Action for Jira is now ready! Users can interact with Jira through natural language, making project management more accessible and efficient. Consider extending the functionality by adding more Jira API endpoints to your OpenAPI schema based on your team's specific needs.

For further customization or to report issues, refer to the [Jira REST API documentation](https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/).