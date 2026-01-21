# Building a GPT Action for Notion: A Step-by-Step Guide

## Introduction

This guide walks you through creating a GPT Action that connects ChatGPT to your Notion workspace. This Action enables users to ask natural language questions and receive answers based on content from your Notion pages. Think of it as creating a knowledgeable librarian for your company's Notion knowledge base.

### Prerequisites

Before you begin, ensure you have:

1. **A Notion workspace** with populated pages containing the information you want to query
2. **Administrator access** to your Notion workspace to create integrations and share pages
3. **A ChatGPT Plus or Enterprise subscription** to create Custom GPTs

## Step 1: Set Up Your Notion Integration

First, you need to create an internal integration in Notion and configure access permissions.

1. **Navigate to Notion Integrations:**
   - Go to [Notion's integrations page](https://www.notion.so/my-integrations)
   - Click "New integration" or "Create new integration"

2. **Configure Your Integration:**
   - Give your integration a name (e.g., "ChatGPT Assistant")
   - Select the workspace where your content resides
   - Choose "Internal integration" (this uses API key authentication)
   - Click "Submit"

3. **Save Your API Key:**
   - After creating the integration, you'll see an "Internal Integration Secret"
   - **Copy this value** - this is your API key (Bearer token)
   - Keep this secure as you'll need it later

4. **Share Content with Your Integration:**
   - For each page, database, or wiki you want the GPT Action to access:
     - Open the page in Notion
     - Click the "•••" menu in the top-right corner
     - Select "Add connections" or "Share with integration"
     - Choose your newly created integration
   - **Important:** The integration can only access content explicitly shared with it

## Step 2: Create Your Custom GPT in ChatGPT

Now, let's configure the Custom GPT that will use your Notion integration.

1. **Access GPT Builder:**
   - Go to [chat.openai.com/gpts/editor](https://chat.openai.com/gpts/editor)
   - Click "Create a GPT"

2. **Configure the Instructions:**
   - In the "Instructions" panel, paste the following:

```markdown
**Context**: You are a helpful chatbot focussed on retrieving information from a company's Notion. An administrator has given you access to a number of useful Notion pages. You are to act similar to a librarian and be helpful answering and finding answers for users' questions.

**Instructions**:
1. Use the search functionality to find the most relevant page or pages.
   - Display the top 3 pages. Include a formatted list containing: Title, Last Edit Date, Author.
   - The Title should be a link to that page.
   1.a. If there are no relevant pages, reword the search and try again (up to 3x)
   1.b. If there are no relevant pages after retries, return "I'm sorry, I cannot find the right info to help you with that question"
2. Open the most relevant article, retrieve and read all of the contents (including any relevant linked pages or databases), and provide a 3 sentence summary. Always provide a quick summary before moving to the next step.
3. Ask the user if they'd like to see more detail. If yes, provide it and offer to explore more relevant pages.

**Additional Notes**: 
- If the user says "Let's get started", introduce yourself as a librarian for the Notion workspace, explain that the user can provide a topic or question, and that you will help to look for relevant pages.
- If there is a database on the page. Always read the database when looking at page contents.
```

3. **Configure the Action Schema:**
   - Click the "Configure" tab
   - In the "Actions" section, click "Create new action"
   - Paste the following OpenAPI schema:

```yaml
openapi: 3.1.0
info:
  title: Notion API
  description: API for interacting with Notion's pages, databases, and users.
  version: 1.0.0
servers:
  - url: https://api.notion.com/v1
    description: Main Notion API server
paths:
  /users:
    get:
      operationId: listAllUsers
      summary: List all users
      parameters:
        - name: Notion-Version
          in: header
          required: true
          schema:
            type: string
          example: 2022-06-28
          constant: 2022-06-28
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      type: object
                      properties:
                        id:
                          type: string
                        name:
                          type: string
                        avatar_url:
                          type: string
                        type:
                          type: string
  /blocks/{block_id}/children:
    get:
      operationId: retrieveBlockChildren
      summary: Retrieve block children
      parameters:
        - name: block_id
          in: path
          required: true
          schema:
            type: string
        - name: Notion-Version
          in: header
          required: true
          schema:
            type: string
          example: 2022-06-28
          constant: 2022-06-28
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  object:
                    type: string
                  results:
                    type: array
                    items:
                      type: object
                      properties:
                        id:
                          type: string
                        type:
                          type: string
                        has_children:
                          type: boolean
  /comments:
    get:
      operationId: retrieveComments
      summary: Retrieve comments
      parameters:
        - name: Notion-Version
          in: header
          required: true
          schema:
            type: string
          example: 2022-06-28
          constant: 2022-06-28
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      type: object
                        properties:
                          id:
                            type: string
                          text:
                            type: string
                          created_time:
                            type: string
                            format: date-time
                          created_by:
                            type: object
                            properties:
                              id:
                                type: string
                              name:
                                type: string
  /pages/{page_id}/properties/{property_id}:
    get:
      operationId: retrievePagePropertyItem
      summary: Retrieve a page property item
      parameters:
        - name: page_id
          in: path
          required: true
          schema:
            type: string
        - name: property_id
          in: path
          required: true
          schema:
            type: string
        - name: Notion-Version
          in: header
          required: true
          schema:
            type: string
          example: 2022-06-28
          constant: 2022-06-28
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                  type:
                    type: string
                  title:
                    type: array
                    items:
                      type: object
                      properties:
                        type:
                          type: string
                        text:
                          type: object
                          properties:
                            content:
                              type: string
  /databases/{database_id}/query:
    post:
      operationId: queryDatabase
      summary: Query a database
      parameters:
        - name: database_id
          in: path
          required: true
          schema:
            type: string
        - name: Notion-Version
          in: header
          required: true
          schema:
            type: string
          example: 2022-06-28
          constant: 2022-06-28
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                filter:
                  type: object
                sorts:
                  type: array
                  items:
                    type: object
                start_cursor:
                  type: string
                page_size:
                  type: integer
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  object:
                    type: string
                  results:
                    type: array
                    items:
                      type: object
                  next_cursor:
                    type: string
                  has_more:
                    type: boolean
  /search:
    post:
      operationId: search
      summary: Search
      parameters:
        - name: Notion-Version
          in: header
          required: true
          schema:
            type: string
          example: 2022-06-28
          constant: 2022-06-28
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                filter:
                  type: object
                  properties:
                    value:
                      type: string
                    property:
                      type: string
                sort:
                  type: object
                  properties:
                    direction:
                      type: string
                    timestamp:
                      type: string
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  object:
                    type: string
                  results:
                    type: array
                    items:
                      type: object
                      properties:
                        id:
                          type: string
                        title:
                          type: array
                          items:
                            type: object
                            properties:
                              type:
                                type: string
                              text:
                                type: object
                                properties:
                                  content:
                                    type: string
```

## Step 3: Configure Authentication

Now, let's connect your Notion integration to the GPT Action.

1. **Set Up API Key Authentication:**
   - In the GPT Action configuration, click "Authentication"
   - Select "API Key" as the authentication method
   - Choose "Bearer" as the Auth Type
   - Paste your "Internal Integration Secret" (API key) from Step 1 into the API Key field

2. **Add Required Headers:**
   - The OpenAPI schema already includes the required `Notion-Version` header
   - ChatGPT will automatically include your Bearer token in the Authorization header

## Step 4: Test Your GPT Action

Now you're ready to test your Notion-connected GPT.

1. **Save and Publish:**
   - Click "Save" in the top-right corner
   - Choose who can access your GPT (Only me, Anyone with a link, or Public)
   - Click "Confirm"

2. **Test the Integration:**
   - Open a chat with your new GPT
   - Start with "Let's get started" to trigger the introduction
   - Ask a question about content in your Notion workspace
   - The GPT should:
     - Search for relevant pages
     - Display the top 3 matches with titles (as links), last edit dates, and authors
     - Provide a summary of the most relevant page
     - Ask if you want more detail

## Troubleshooting Common Issues

### Issue: Search Returns No Results
**Solution:** Double-check that you've shared the relevant pages with your integration in Notion. The integration can only access content explicitly shared with it.

### Issue: Authentication Errors
**Solution:** Verify that:
- Your API key (Internal Integration Secret) is correctly copied
- The authentication is set to "Bearer" type
- Your integration is still active in Notion

### Issue: Missing Content in Results
**Solution:** Ensure that:
- Pages are properly populated with content
- Databases on pages are also shared with the integration
- The content is in a supported format (text, databases, etc.)

## Best Practices for Organization

To get the best results from your Notion GPT Action:

1. **Organize Content in Wikis:** Notion works best with well-structured wikis. Consider organizing knowledge base content into dedicated wikis.

2. **Use Descriptive Titles:** Page titles should clearly indicate their content to improve search accuracy.

3. **Regular Maintenance:** Periodically review which pages are shared with the integration and update as your knowledge base evolves.

4. **Test with Real Questions:** Have team members test the GPT with actual questions they would ask to ensure it provides useful answers.

## Next Steps

Your Notion GPT Action is now ready to help users find information quickly and naturally. Consider:

1. **Expanding Access:** Share the GPT with team members who need to query your Notion knowledge base
2. **Gathering Feedback:** Ask users what questions they're asking and whether they're getting useful answers
3. **Iterating:** Based on feedback, refine your instructions or share additional pages with the integration

Remember that this integration uses Notion's internal integration approach. If you need OAuth authentication for a public-facing application, review [Notion's authorization documentation](https://developers.notion.com/docs/authorization) to determine the best approach for your use case.