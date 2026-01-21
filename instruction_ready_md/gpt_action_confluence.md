# Confluence GPT Action: Developer Cookbook

## Introduction

This guide walks you through building a GPT Action that connects to Atlassian Confluence. This Action enables ChatGPT to search your organization's Confluence wiki and retrieve information to answer user questions.

### Prerequisites

Before you begin, ensure you have:
- An Atlassian account with permissions to create apps in the [Atlassian Developer Console](https://developer.atlassian.com/console/myapps/)
- A Confluence instance with content you want to query
- Familiarity with [GPT Actions](https://platform.openai.com/docs/actions) and the [Actions Library](https://platform.openai.com/docs/actions/actions-library)

## Step 1: Create Your Confluence OAuth App

First, you need to create an OAuth 2.0 integration in the Atlassian Developer Portal.

1. Navigate to the [Atlassian Developer Console](https://developer.atlassian.com/console/myapps/)
2. Click **Create** and select **OAuth 2.0 integration**
3. Provide a name for your app, agree to the terms, and click **Create**
4. In the left sidebar, select **Distribution** and click **Edit**
5. Change the distribution setting to **Sharing**
6. Fill in the required fields and click **Save Changes**
7. Select **Permissions** from the left sidebar
8. Add the following scopes:
   - `read:confluence-content.all`
   - `search:confluence`
   - `read:me`
9. Select **Authorization** from the left sidebar
10. Click **Add** under "Action" in the OAuth 2.0 row
11. You'll need to add a callback URL here later—for now, use a placeholder like `https://example.com/callback`
12. Select **Settings** from the left sidebar
13. **Copy your Client ID and Client Secret**—you'll need these for the GPT Action setup

## Step 2: Configure Your Custom GPT

Now, create and configure your Custom GPT in ChatGPT.

### 2.1 Add Custom Instructions

In your Custom GPT's **Instructions** panel, paste the following:

```
You are a "Confluence Savant", equipped with the ability to search our company's Product Wiki in Confluence to answer product-related questions.

You must ALWAYS perform the "getAccessibleResources" Action first to get the "cloudid" value you will need in subsequent Actions.

Your job is to provide accurate and detailed responses by retrieving information from the Product Wiki. Your responses should be clear, concise, and directly address the question asked. You have the capability to execute an action named "performConfluenceSearch" that allows you to search for content within our Confluence Product Wiki using specific terms or phrases related to the user's question.

    - When you receive a query about product information, use the "performConfluenceSearch" action to retrieve relevant content from the Product Wiki. Formulate your search query based on the user's question, using specific keywords or phrases to find the most pertinent information.
    - Once you receive the search results, review the content to ensure it matches the user's query. If necessary, refine your search query to retrieve more accurate results.
    - Provide a response that synthesizes the information from the Product Wiki, clearly answering the user's question. Your response should be easy to understand and directly related to the query.
    - If the query is complex or requires clarification, ask follow-up questions to the user to refine your understanding and improve the accuracy of your search.
    - If the information needed to answer the question is not available in the Product Wiki, inform the user and guide them to where they might find the answer, such as contacting a specific department or person in the company.

    Here is an example of how you might respond to a query:

    User: "What are the latest features of our XYZ product?"
    You: "The latest features of the XYZ product, as detailed in our Product Wiki, include [feature 1], [feature 2], and [feature 3]. These features were added in the recent update to enhance [specific functionalities]. For more detailed information, you can refer to the Product Wiki page [link to the specific Confluence page]."

Remember, your goal is to provide helpful, accurate, and relevant information to the user's query by effectively leveraging the Confluence Product Wiki.
```

### 2.2 Add the OpenAPI Schema

In your Custom GPT's **Actions** panel, paste the following OpenAPI schema:

```yaml
openapi: 3.1.0
info:
  title: Atlassian API
  description: This API provides access to Atlassian resources through OAuth token authentication.
  version: 1.0.0
servers:
  - url: https://api.atlassian.com
    description: Main API server
paths:
  /oauth/token/accessible-resources:
    get:
      operationId: getAccessibleResources
      summary: Retrieves accessible resources for the authenticated user.
      description: This endpoint retrieves a list of resources the authenticated user has access to, using an OAuth token.
      security:
        - bearerAuth: []
      responses:
        '200':
          description: A JSON array of accessible resources.
          content:
            application/json:
              schema: 
                $ref: '#/components/schemas/ResourceArray'
  /ex/confluence/{cloudid}/wiki/rest/api/search:
    get:
      operationId: performConfluenceSearch
      summary: Performs a search in Confluence based on a query.
      description: This endpoint allows searching within Confluence using the CQL (Confluence Query Language).
      parameters:
        - in: query
          name: cql
          required: true
          description: The Confluence Query Language expression to evaluate.
          schema:
            type: string
        - in: path
          name: cloudid
          required: true
          schema:
            type: string
          description: The cloudid retrieved from the getAccessibleResources Action
        - in: query
          name: cqlcontext
          description: The context to limit the search, specified as JSON.
          schema:
            type: string
        - in: query
          name: expand
          description: A comma-separated list of properties to expand on the search result.
          schema:
            type: string
      responses:
        '200':
          description: A list of search results matching the query.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SearchResults'
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  schemas:
    ResourceArray:
      type: array
      items:
        $ref: '#/components/schemas/Resource'
    Resource:
      type: object
      required:
        - id
        - name
        - type
      properties:
        id:
          type: string
          description: The unique identifier for the resource.
        name:
          type: string
          description: The name of the resource.
        type:
          type: string
          description: The type of the resource.
    SearchResults:
      type: object
      properties:
        results:
          type: array
          items:
            $ref: '#/components/schemas/SearchResult'
    SearchResult:
      type: object
      properties:
        id:
          type: string
          description: The unique identifier of the content.
        title:
          type: string
          description: The title of the content.
        type:
          type: string
          description: The type of the content (e.g., page, blog post).
        space:
          type: object
          properties:
            id:
              type: string
              description: The space ID where the content is located.
            name:
              type: string
              description: The name of the space.
```

## Step 3: Configure Authentication

Now, set up OAuth authentication in your Custom GPT.

1. In your Custom GPT's **Authentication** section, select **OAuth**
2. Enter the following details:
   - **Client ID**: Paste the Client ID you copied from the Atlassian Developer Console
   - **Client Secret**: Paste the Client Secret you copied
   - **Authorization URL**: `https://auth.atlassian.com/authorize`
   - **Token URL**: `https://auth.atlassian.com/oauth/token`
   - **Scope**: `read:confluence-content.all search:confluence`
   - **Token**: Keep as **Default (POST)**

## Step 4: Finalize the OAuth Configuration

After setting up authentication in ChatGPT, you need to complete the OAuth configuration in the Atlassian Developer Console.

1. **Copy the callback URL** generated by ChatGPT in the Authentication section
2. Return to your app in the Atlassian Developer Console
3. Navigate to **Authorization** in the left sidebar
4. Replace the placeholder callback URL with the actual callback URL from ChatGPT
5. Save your changes

## How It Works

When a user asks your GPT a question:

1. The GPT first calls `getAccessibleResources` to retrieve the user's accessible Confluence resources and obtain the `cloudid`
2. Using the `cloudid`, the GPT calls `performConfluenceSearch` with a CQL query based on the user's question
3. The GPT processes the search results and synthesizes a natural language response
4. The response is presented to the user with relevant information from Confluence

## Troubleshooting

### Callback URL Error
If you encounter a callback URL error in ChatGPT, ensure you've added the exact callback URL from ChatGPT to your Confluence app's **Authorized redirect URIs** in the Atlassian Developer Console.

### Incorrect Project or Dataset
If ChatGPT searches the wrong Confluence space or dataset, update your Custom GPT instructions to be more specific about which Confluence space or project to search.

### Looping Actions
If the GPT gets stuck in a loop of calling Actions, verify that you've granted all necessary scopes (`read:confluence-content.all`, `search:confluence`) to your OAuth app.

### Authentication Failures
If authentication fails, double-check:
- Your Client ID and Client Secret are correctly copied
- The scopes match exactly what's configured in the Atlassian Developer Console
- The callback URL is correctly configured in both ChatGPT and the Atlassian Developer Console

## Next Steps

Your Confluence GPT Action is now ready to use! Test it by asking questions about content in your Confluence instance. For more advanced functionality, you can extend the OpenAPI schema to include additional Confluence API endpoints for creating or updating content.