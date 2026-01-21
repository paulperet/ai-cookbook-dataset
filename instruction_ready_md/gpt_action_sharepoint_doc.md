# Building a GPT Action for SharePoint File Retrieval

## Introduction

This guide walks you through building a GPT Action that enables ChatGPT to search and retrieve files from SharePoint or Office365. The solution uses Microsoft Graph API to search for files and an Azure Function middleware to format the response for ChatGPT's file handling capabilities.

### Prerequisites

Before starting, ensure you have:
- Access to a SharePoint/Office365 environment
- Basic knowledge of APIs and OAuth
- An Azure account for hosting the middleware function
- Familiarity with [GPT Actions](https://platform.openai.com/docs/actions)

## Architecture Overview

The solution follows this workflow:
1. User asks ChatGPT a question
2. ChatGPT calls your Azure Function with a search term
3. Azure Function authenticates with Microsoft Graph API
4. Function searches SharePoint for relevant files
5. Function retrieves file contents and converts to base64
6. Function returns formatted response to ChatGPT
7. ChatGPT uses the files as if uploaded directly

## Step 1: Set Up Azure Function

First, create an Azure Function to serve as middleware:

1. **Create Function App**: Follow the [Azure Function cookbook](https://cookbook.openai.com/examples/chatgpt/gpt_actions_library/gpt_middleware_azure_function) to set up a Node.js Azure Function

2. **Configure Authentication**: Set up Microsoft Entra ID (formerly Azure AD) authentication for your function

## Step 2: Install Required Packages

In your Azure Function console, install the necessary packages:

```bash
npm install @microsoft/microsoft-graph-client
npm install axios
```

## Step 3: Configure Environment Variables

Set the following environment variables in your Azure Function Configuration:

```bash
TENANT_ID=your_tenant_id
CLIENT_ID=your_client_id
MICROSOFT_PROVIDER_AUTHENTICATION_SECRET=your_client_secret
```

## Step 4: Implement the Azure Function

Create a new function file and add the following implementation:

### 4.1 Import Required Modules

```javascript
const { Client } = require('@microsoft/microsoft-graph-client');
const axios = require('axios');
const qs = require('querystring');
```

### 4.2 Create Authentication Helper Functions

First, create a function to initialize the Microsoft Graph client:

```javascript
function initGraphClient(accessToken) {
    return Client.init({
        authProvider: (done) => {
            done(null, accessToken);
        }
    });
}
```

Next, implement the On-Behalf-Of (OBO) token retrieval:

```javascript
async function getOboToken(userAccessToken) {
    const { TENANT_ID, CLIENT_ID, MICROSOFT_PROVIDER_AUTHENTICATION_SECRET } = process.env;
    const params = {
        client_id: CLIENT_ID,
        client_secret: MICROSOFT_PROVIDER_AUTHENTICATION_SECRET,
        grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        assertion: userAccessToken,
        requested_token_use: 'on_behalf_of',
        scope: 'https://graph.microsoft.com/.default'
    };

    const url = `https://login.microsoftonline.com/${TENANT_ID}/oauth2/v2.0/token`;
    try {
        const response = await axios.post(url, qs.stringify(params), {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        return response.data.access_token;
    } catch (error) {
        console.error('Error obtaining OBO token:', error.response?.data || error.message);
        throw error;
    }
}
```

### 4.3 Implement File Content Retrieval

Create a function to fetch and format file contents:

```javascript
async function getDriveItemContent(client, driveId, itemId, name) {
    try {
        const filePath = `/drives/${driveId}/items/${itemId}`;
        const downloadPath = filePath + `/content`;
        
        // Get file content as stream
        const fileStream = await client.api(downloadPath).getStream();
        
        // Convert stream to base64
        let chunks = [];
        for await (let chunk of fileStream) {
            chunks.push(chunk);
        }
        const base64String = Buffer.concat(chunks).toString('base64');
        
        // Get file metadata
        const file = await client.api(filePath).get();
        const mime_type = file.file.mimeType;
        const fileName = file.name;
        
        return {
            "name": fileName,
            "mime_type": mime_type,
            "content": base64String
        };
    } catch (error) {
        console.error('Error fetching drive content:', error);
        throw new Error(`Failed to fetch content for ${name}: ${error.message}`);
    }
}
```

### 4.4 Create Main Azure Function

Implement the main function that orchestrates the entire process:

```javascript
module.exports = async function (context, req) {
    // Extract search term from request
    const searchTerm = req.query.searchTerm || (req.body && req.body.searchTerm);
    
    // Validate authorization header
    if (!req.headers.authorization) {
        context.res = {
            status: 400,
            body: 'Authorization header is missing'
        };
        return;
    }
    
    // Extract and validate bearer token
    const bearerToken = req.headers.authorization.split(' ')[1];
    let accessToken;
    try {
        accessToken = await getOboToken(bearerToken);
    } catch (error) {
        context.res = {
            status: 500,
            body: `Failed to obtain OBO token: ${error.message}`
        };
        return;
    }
    
    // Initialize Graph client
    let client = initGraphClient(accessToken);
    
    // Construct search request body
    const requestBody = {
        requests: [
            {
                entityTypes: ['driveItem'],
                query: {
                    queryString: searchTerm
                },
                from: 0,
                size: 10  // Limit to top 10 results
            }
        ]
    };
    
    try {
        // Execute search
        const list = await client.api('/search/query').post(requestBody);
        
        // Process search results
        const processList = async () => {
            const results = [];
            await Promise.all(list.value[0].hitsContainers.map(async (container) => {
                for (const hit of container.hits) {
                    if (hit.resource["@odata.type"] === "#microsoft.graph.driveItem") {
                        const { name, id } = hit.resource;
                        const driveId = hit.resource.parentReference.driveId;
                        const contents = await getDriveItemContent(client, driveId, id, name);
                        results.push(contents);
                    }
                }
            }));
            return results;
        };
        
        let results;
        if (list.value[0].hitsContainers[0].total == 0) {
            results = 'No results found';
        } else {
            results = await processList();
            results = { 'openaiFileResponse': results };
        }
        
        context.res = {
            status: 200,
            body: results
        };
    } catch (error) {
        context.res = {
            status: 500,
            body: `Error performing search or processing results: ${error.message}`,
        };
    }
};
```

## Step 5: Test Your Function

Test your Azure Function using Postman or curl:

```bash
curl -X POST \
  https://your-function-app.azurewebsites.net/api/your-function-name \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"searchTerm": "quarterly report"}'
```

## Step 6: Configure Custom GPT

### 6.1 Add Instructions

In your Custom GPT's Instructions panel, add:

```
You are a Q&A helper that helps answer users questions. You have access to a documents repository through your API action. When a user asks a question, you pass in the "searchTerm" a single keyword or term you think you should use for the search.

****

Scenario 1: There are answers

If your action returns results, then you take the results from the action and try to answer the users question. 

****

Scenario 2: No results found

If the response you get from the action is "No results found", stop there and let the user know there were no results and that you are going to try a different search term, and explain why. You must always let the user know before conducting another search.

Example:

****

I found no results for "DEI". I am now going to try [insert term] because [insert explanation]

****

Then, try a different searchTerm that is similar to the one you tried before, with a single word. 

Try this three times. After the third time, then let the user know you did not find any relevant documents to answer the question, and to check SharePoint. 
Be sure to be explicit about what you are searching for at each step.

****

In either scenario, try to answer the user's question. If you cannot answer the user's question based on the knowledge you find, let the user know and ask them to go check the HR Docs in SharePoint. 
```

### 6.2 Configure OpenAPI Schema

In the Actions panel, add this OpenAPI schema (update placeholders):

```yaml
openapi: 3.1.0
info:
  title: SharePoint Search API
  description: API for searching SharePoint documents.
  version: 1.0.0
servers:
  - url: https://{your_function_app_name}.azurewebsites.net/api
    description: SharePoint Search API server
paths:
  /{your_function_name}:
    post:
      operationId: searchSharePoint
      summary: Searches SharePoint for documents matching a query and term.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                searchTerm:
                  type: string
                  description: A specific term to search for within the documents.
      responses:
        '200':
          description: A CSV file of query results encoded in base64.
          content:
            application/json:
              schema:
                type: object
                properties:
                  openaiFileResponseData:
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
                          description: The base64 encoded contents of the file.
        '400':
          description: Bad request when the SQL query parameter is missing.
        '413':
          description: Payload too large if the response exceeds the size limit.
        '500':
          description: Server error when there are issues executing the query or encoding the results.
```

## Step 7: Customization Options

### 7.1 Limit Search Scope

Modify the search query to target specific SharePoint sites:

```javascript
// Add site filtering to requestBody
const requestBody = {
    requests: [
        {
            entityTypes: ['driveItem'],
            query: {
                queryString: `site:your-site-name ${searchTerm}`
            },
            from: 0,
            size: 10
        }
    ]
};
```

### 7.2 Filter File Types

Restrict to specific file types (e.g., only PDFs):

```javascript
// In the processList function, add file type check
if (hit.resource["@odata.type"] === "#microsoft.graph.driveItem") {
    const fileType = hit.resource.name.split('.').pop().toLowerCase();
    if (fileType === 'pdf') {  // Only process PDFs
        const { name, id } = hit.resource;
        const driveId = hit.resource.parentReference.driveId;
        const contents = await getDriveItemContent(client, driveId, id, name);
        results.push(contents);
    }
}
```

### 7.3 Adjust Result Count

Change the number of files returned (maximum 10 per OpenAI guidelines):

```javascript
// In requestBody
size: 5  // Return only top 5 results
```

## Considerations and Limitations

1. **File Size Limits**: OpenAI Actions have a 100K character limit per response
2. **Timeout**: Actions must complete within 45 seconds
3. **Authentication**: Ensure proper OAuth flow for user context
4. **Rate Limiting**: Microsoft Graph API has rate limits
5. **File Types**: Some file types may not be supported by ChatGPT

## Testing Your Implementation

1. **Test Authentication**: Verify OBO token flow works correctly
2. **Test Search**: Ensure search returns relevant files
3. **Test File Retrieval**: Confirm files are properly converted to base64
4. **Test GPT Integration**: Verify ChatGPT can use returned files
5. **Test Error Handling**: Ensure graceful failure for missing files or permissions

## Troubleshooting

- **Authentication Errors**: Check tenant ID, client ID, and secret configuration
- **No Results**: Verify user has access to SharePoint files
- **Timeout Issues**: Reduce the number of files retrieved or implement pagination
- **Format Errors**: Ensure response matches OpenAI's file response structure

## Next Steps

Once your basic implementation is working, consider:
1. Adding caching for frequently accessed files
2. Implementing file chunking for large documents
3. Adding search result ranking or relevance scoring
4. Creating a user interface for managing search preferences
5. Adding support for additional file repositories

Your GPT Action is now ready to help users search and retrieve files from SharePoint using natural language queries!