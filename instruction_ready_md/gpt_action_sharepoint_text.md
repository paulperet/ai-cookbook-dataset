# GPT Action Library: SharePoint (Return as Document) - Cookbook Guide

## Introduction

This guide provides a step-by-step tutorial for building a GPT Action that enables ChatGPT to answer user questions using the context of files they can access in SharePoint or Office 365. The solution leverages Microsoft's Graph API search capabilities and file retrieval, with an Azure Function middleware layer to process and convert file contents into a format ChatGPT can understand.

### Key Value Proposition
Users can leverage ChatGPT's natural language capabilities to query and analyze content from their SharePoint documents directly through natural conversation.

**Example Use Cases:**
- Finding which files relate to a specific topic
- Extracting answers to critical questions buried deep within documents

## Prerequisites

Before starting, ensure you have:

1. **Access to a SharePoint environment** with files to search
2. **Postman** installed and basic knowledge of APIs and OAuth flows
3. **An OpenAI API Key** from [platform.openai.com](https://platform.openai.com)
4. **Azure Account** with Function App creation permissions

## Architecture Overview

This solution follows a four-step process:

1. **Search**: Find relevant files the user can access based on their question
2. **Retrieve**: Download and convert files to readable text
3. **Analyze**: Use GPT-4o-mini to extract relevant text snippets from the documents
4. **Respond**: Return processed information to ChatGPT for final answer generation

The key difference from simpler solutions is the preprocessing step that converts files to text and summarizes content before sending to ChatGPT, enabling analysis of larger, unstructured documents.

## Step 1: Set Up the Azure Function

First, create an Azure Function to serve as middleware. If you're new to Azure Functions, follow the [Azure Function cookbook](https://cookbook.openai.com/examples/chatgpt/gpt_actions_library/gpt_middleware_azure_function) for detailed setup instructions.

Once your function is created and authenticated, proceed to the next step.

## Step 2: Configure Environment Variables

Navigate to your Azure Function's **Configuration** tab (under **Settings**) and add the following environment variables:

```bash
TENANT_ID=<your-azure-tenant-id>
CLIENT_ID=<your-azure-client-id>
MICROSOFT_PROVIDER_AUTHENTICATION_SECRET=<your-client-secret>
OPENAI_API_KEY=<your-openai-api-key>
```

You can obtain these values from your Azure Active Directory application registration.

## Step 3: Install Required Dependencies

In your Azure Function's **Console** tab (under **Development Tools**), install the necessary npm packages:

```bash
npm install @microsoft/microsoft-graph-client
npm install axios
npm install pdf-parse
npm install openai
```

These packages provide:
- Microsoft Graph API client functionality
- HTTP request capabilities
- PDF text extraction
- OpenAI API integration

## Step 4: Implement the Azure Function Code

Replace your function's code with the following implementation. This code handles authentication, file search, content extraction, and GPT analysis.

### 4.1 Import Required Modules

```javascript
const { Client } = require('@microsoft/microsoft-graph-client');
const axios = require('axios');
const qs = require('querystring');
const pdfParse = require('pdf-parse');
const path = require('path');
const { OpenAI } = require('openai');
```

### 4.2 Implement Authentication Helper Functions

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

Next, implement the On-Behalf-Of (OBO) token exchange to pass user credentials through to the Graph API:

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

This function handles downloading and converting various file types to extract text content:

```javascript
const getDriveItemContent = async (client, driveId, itemId, name) => {
    try {
        const fileType = path.extname(name).toLowerCase();
        // Supported file types that can be converted to PDF for text extraction
        const allowedFileTypes = ['.pdf', '.doc', '.docx', '.odp', '.ods', '.odt', '.pot', '.potm', '.potx', '.pps', '.ppsx', '.ppsxm', '.ppt', '.pptm', '.pptx', '.rtf'];
        
        // Add format parameter for non-PDF files to convert them
        const filePath = `/drives/${driveId}/items/${itemId}/content` + 
            ((fileType === '.pdf' || fileType === '.txt' || fileType === '.csv') ? '' : '?format=pdf');
        
        if (allowedFileTypes.includes(fileType)) {
            const response = await client.api(filePath).getStream();
            
            // Combine stream chunks into a buffer
            let chunks = [];
            for await (let chunk of response) {
                chunks.push(chunk);
            }
            let buffer = Buffer.concat(chunks);
            
            // Extract text from PDF
            const pdfContents = await pdfParse(buffer);
            return pdfContents.text;
            
        } else if (fileType === '.txt') {
            // Direct text file retrieval
            const response = await client.api(filePath).get();
            return response;
            
        } else if (fileType === '.csv') {
            // CSV file handling
            const response = await client.api(filePath).getStream();
            let chunks = [];
            for await (let chunk of response) {
                chunks.push(chunk);
            }
            let buffer = Buffer.concat(chunks);
            let dataString = buffer.toString('utf-8');
            return dataString;
            
        } else {
            return 'Unsupported File Type';
        }
    } catch (error) {
        console.error('Error fetching drive content:', error);
        throw new Error(`Failed to fetch content for ${name}: ${error.message}`);
    }
};
```

### 4.4 Implement GPT Text Analysis

This function uses GPT-4o-mini to extract only the relevant parts of document text based on the user's query:

```javascript
const getRelevantParts = async (text, query) => {
    try {
        const openAIKey = process.env["OPENAI_API_KEY"];
        const openai = new OpenAI({
            apiKey: openAIKey,
        });
        
        const response = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that finds relevant content in text based on a query. You only return the relevant sentences, and you return a maximum of 10 sentences"
                },
                {
                    "role": "user", 
                    "content": `Based on this question: **"${query}"**, get the relevant parts from the following text:*****\n\n${text}*****. If you cannot answer the question based on the text, respond with 'No information provided'`
                }
            ],
            temperature: 0, // Deterministic extraction
            max_tokens: 1000 // Adjust based on your document volume
        });
        
        return response.choices[0].message.content;
    } catch (error) {
        console.error('Error with OpenAI:', error);
        return 'Error processing text with OpenAI' + error;
    }
};
```

### 4.5 Implement the Main Azure Function

Now, combine all helper functions into the main Azure Function that orchestrates the entire workflow:

```javascript
module.exports = async function (context, req) {
    // Extract query parameters
    const query = req.query.query || (req.body && req.body.query);
    const searchTerm = req.query.searchTerm || (req.body && req.body.searchTerm);
    
    // Validate authorization header
    if (!req.headers.authorization) {
        context.res = {
            status: 400,
            body: 'Authorization header is missing'
        };
        return;
    }
    
    // Extract and exchange bearer token for OBO token
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
    
    // Configure search request
    const requestBody = {
        requests: [
            {
                entityTypes: ['driveItem'],
                query: {
                    queryString: searchTerm
                },
                from: 0,
                size: 10 // Adjust based on your needs
            }
        ]
    };

    try {
        // Helper functions for text processing
        const tokenizeContent = (content) => {
            return content.split(/\s+/);
        };

        const breakIntoTokenWindows = (tokens) => {
            const tokenWindows = [];
            const maxWindowTokens = 10000; // 10k tokens per window
            let startIndex = 0;

            while (startIndex < tokens.length) {
                const window = tokens.slice(startIndex, startIndex + maxWindowTokens);
                tokenWindows.push(window);
                startIndex += maxWindowTokens;
            }

            return tokenWindows;
        };
        
        // Execute search
        const list = await client.api('/search/query').post(requestBody);
        
        // Process search results
        const processList = async () => {
            const results = [];

            await Promise.all(list.value[0].hitsContainers.map(async (container) => {
                for (const hit of container.hits) {
                    if (hit.resource["@odata.type"] === "#microsoft.graph.driveItem") {
                        const { name, id } = hit.resource;
                        const webUrl = hit.resource.webUrl.replace(/\s/g, "%20");
                        const rank = hit.rank;
                        const driveId = hit.resource.parentReference.driveId;
                        
                        // Retrieve file content
                        const contents = await getDriveItemContent(client, driveId, id, name);
                        
                        if (contents !== 'Unsupported File Type') {
                            // Tokenize and process large documents in chunks
                            const tokens = tokenizeContent(contents);
                            const tokenWindows = breakIntoTokenWindows(tokens);
                            
                            // Analyze each chunk with GPT
                            const relevantPartsPromises = tokenWindows.map(
                                window => getRelevantParts(window.join(' '), query)
                            );
                            const relevantParts = await Promise.all(relevantPartsPromises);
                            const combinedResults = relevantParts.join('\n');
                            
                            results.push({ name, webUrl, rank, contents: combinedResults });
                        } else {
                            results.push({ name, webUrl, rank, contents: 'Unsupported File Type' });
                        }
                    }
                }
            }));

            return results;
        };
        
        // Execute processing and return results
        const processedResults = await processList();
        
        // Sort by relevance rank
        processedResults.sort((a, b) => a.rank - b.rank);
        
        context.res = {
            status: 200,
            body: processedResults
        };
        
    } catch (error) {
        console.error('Error in function execution:', error);
        context.res = {
            status: 500,
            body: `Function error: ${error.message}`
        };
    }
};
```

## Step 5: Test Your Function

After saving your function code, test it using Postman:

1. Make a POST request to your Azure Function URL
2. Include the Authorization header with a valid bearer token
3. Send a JSON body with your test parameters:

```json
{
    "query": "What are the Q3 sales figures?",
    "searchTerm": "sales report Q3"
}
```

If successful, you should receive a response containing processed document information with relevant text extracts.

## Step 6: Integrate with Custom GPT

Once your function is working correctly:

1. Create a new Custom GPT in the ChatGPT interface
2. Add your Azure Function as an Action
3. Configure the authentication settings to match your Azure AD application
4. Define the action schema to match your function's expected input/output

## Customization Tips

### Adjusting Search Parameters
- Modify the `size` parameter in the search request body to control how many results are returned
- Adjust the `from` parameter for pagination support

### Optimizing Text Processing
- Change the `max_tokens` parameter in the GPT call based on your typical document sizes
- Adjust the token window size (currently 10,000) for different GPT model limits

### Supporting Additional File Types
- Extend the `allowedFileTypes` array to include other Office file formats
- Implement additional text extraction logic for specific file types

## Troubleshooting

### Common Issues and Solutions

1. **Authentication Errors**: Ensure your Azure AD application has the correct API permissions (Files.Read.All, Sites.Read.All) and that the client secret is valid.

2. **File Conversion Failures**: Verify that the files you're testing with are in supported formats and not password-protected.

3. **Timeout Errors**: Large documents may cause timeouts. Consider reducing the `size` parameter or implementing more aggressive text summarization.

4. **GPT Analysis Errors**: Check your OpenAI API key and ensure you have sufficient credits. Monitor token usage to avoid exceeding limits.

## Conclusion

You've now built a complete GPT Action that enables ChatGPT to search, retrieve, and analyze content from SharePoint documents. This solution provides a powerful way to leverage existing document repositories for AI-powered question answering while maintaining security through proper authentication and authorization flows.

Remember that this code is meant to be directional—customize it to fit your specific requirements, document structures, and security policies.