# GPT Action Library: Sharepoint (Return as Document)

## Introduction

This page provides an instruction & guide for developers building a GPT Action for a specific application. Before you proceed, make sure to first familiarize yourself with the following information: 
- [Introduction to GPT Actions](https://platform.openai.com/docs/actions)
- [Introduction to GPT Actions Library](https://platform.openai.com/docs/actions/actions-library)
- [Example of Building a GPT Action from Scratch](https://platform.openai.com/docs/actions/getting-started)

This solution enables a GPT action to answer a user’s question with the context of files the user can access in SharePoint or Office365, using Microsoft’s Graph API [search capabilities](https://learn.microsoft.com/en-us/graph/api/resources/search-api-overview?view=graph-rest-1.0) and the ability to [retrieve files](https://learn.microsoft.com/en-us/graph/api/driveitem-get?view=graph-rest-1.0\&tabs=http). It uses Azure Functions to process the Graph API response and convert it to a human readable format or structure it in a way ChatGPT understands. This code is meant to be directional, and you should modify it to your requirements.

This solution pre-processes the file within the Azure Function. The Azure Function returns text, instead of the base64 encoded file. Due to the pre-processing and the conversion to text, this solution is best used for large, unstructured documents, and for when you want to analyze more than the amount of files supported in the first solution (see documentation [here](https://platform.openai.com/docs/actions/sending-files)).

### Value + Example Business Use Cases

**Value**: Users can now leverage ChatGPT's natural language capability to connect directly to files in Sharpeoint

**Example Use Cases**: 
- A user needs to look up which files relate to a certain topic
- A user needs an answer to a critical question, buried deep in documents

## Architecture / Example

This solution uses a Node.js Azure Function to, based on the logged in user:

1. Search for a relevant file that the user has access to, based on the user’s initial question.

2. For each file that is found, convert it to a consistent readable format and retrieve all the text.

3. Use GPT 4o mini (gpt-4o-mini) to extract the relevant text from the files based on the initial user’s question. Note the pricing of GPT 4o mini [here](https://openai.com/pricing#language-models) - since we are dealing with small token chunks, the cost of this step is nominal.  

4. Returns that data to ChatGPT. The GPT then uses that information to respond to the user's initial question.

As you can see from the below architecture diagram, the first three steps are the same as Solution 1. The main difference is that this solution converts the file to text instead of a base64 string, and then summarizes that text using GPT 4o mini.

## Application Information

### Application Key Links

Check out these links from the application before you get started:
- Application Website: https://www.microsoft.com/en-us/microsoft-365/sharepoint/collaboration
- Application API Documentation: https://learn.microsoft.com/en-us/previous-versions/office/developer/sharepoint-rest-reference/

### Application Prerequisites

Before you get started, make sure you go through the following steps in your application environment:
- Access to a Sharepoint environment 
- Postman (and knowledge of APIs and OAuth)
- An OpenAI API Key from platform.openai.com

## Middleware Information

If you follow the [search concept files guide](https://learn.microsoft.com/en-us/graph/search-concept-files), the [Microsoft Graph Search API](https://learn.microsoft.com/en-us/graph/search-concept-files) returns references to files that fit the criteria, but not the file contents themselves. Therefore, middleware is required, rather than hitting the MSFT endpoints directly.

Steps: 

1. loop through the returned files and download the files using the [Download File endpoint](https://learn.microsoft.com/en-us/graph/api/driveitem-get-content?view=graph-rest-1.0\&tabs=http) or [Convert File endpoint](https://learn.microsoft.com/en-us/graph/api/driveitem-get-content-format?view=graph-rest-1.0\&tabs=http)

2. convert that Binary stream to human readable text using [pdf-parse](https://www.npmjs.com/package/pdf-parse)

3. Then, we can optimize further by summarizing using gpt-4o-mini in the function to help with the 100,000 character limit we impose on Actions today. 

### Additional Steps

#### Set up Azure Function

1. Set up an Azure Function using the steps in the [Azure Function cookbook](https://cookbook.openai.com/examples/chatgpt/gpt_actions_library/gpt_middleware_azure_function)

#### Add in Function Code

Now that you have an authenticated Azure Function, we can update the function to search SharePoint / O365

2. Go to your test function and paste in the code from [this file](https://github.com/openai/openai-cookbook/blob/main/examples/chatgpt/sharepoint_azure_function/solution_two_preprocessing.js). Save the function.

> **This code is meant to be directional** - while it should work out of the box, it is designed to be customized to your needs (see examples towards the end of this document).

3. Set up the following env variables by going to the **Configuration** tab on the left under **Settings.** Note that this may be listed directly in **Environment Variables** depending on your Azure UI.

    1. `TENANT_ID`: copied from previous section

    2. `CLIENT_ID`: copied from previous section 

    3. `OPENAI_API_KEY:` spin up an OpenAI API key on platform.openai.com.

4. Go to the **Console** tab under the **Development Tools**

    1. Install the following packages in console

       1. `npm install @microsoft/microsoft-graph-client`

       2. `npm install axios`

       3. `npm install pdf-parse`

       4. `npm install openai`

5. Once this is complete, try calling the function (POST call) from Postman again, putting the below into body (using a query and search term you think will generate responses).

    ```json
    {
        "query": "<choose a question>",
        "searchTerm": "<choose a search term>"
    }
    ```

6. If you get a response, you are ready to set this up with a Custom GPT!


## Detailed Walkthrough

The below walks through setup instructions and walkthrough unique to this solution of pre-processing the files and extracting summaries in the Azure Function. You can find the entire code [here](https://github.com/openai/openai-cookbook/blob/main/examples/chatgpt/sharepoint_azure_function/solution_two_preprocessing.js).

### Code Walkthrough

#### Implementing the Authentication 

Below we have a few helper functions that we’ll use in the function.

#### Initializing the Microsoft Graph Client
Create a function to initialize the Graph client with an access token. This will be used to search through Office 365 and SharePoint.

```javascript
const { Client } = require('@microsoft/microsoft-graph-client');

function initGraphClient(accessToken) {
    return Client.init({
        authProvider: (done) => {
            done(null, accessToken);
        }
    });
}
```

#### Obtaining an On-Behalf-Of (OBO) Token
This function uses an existing bearer token to request an OBO token from Microsoft's identity platform. This enables passing through the credentials to ensure the search only returns files the logged-in user can access.

```javascript
const axios = require('axios');
const qs = require('querystring');

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

    const url = `https\://login.microsoftonline.com/${TENANT_ID}/oauth2/v2.0/token`;
    try {
        const response = await axios.post(url, qs.stringify(params), {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        return response.data.access\_token;
    } catch (error) {
        console.error('Error obtaining OBO token:', error.response?.data || error.message);
        throw error;
    }
}
```

#### Retrieving Content from O365 / SharePoint Items

This function fetches the content of drive items, handling different file types and converting files to PDF when necessary for text extraction. This uses the [download endpoint](https://learn.microsoft.com/en-us/graph/api/driveitem-get-content?view=graph-rest-1.0\&tabs=http) for PDFs and the [convert endpoint](https://learn.microsoft.com/en-us/graph/api/driveitem-get-content-format?view=graph-rest-1.0\&tabs=http) for other supported file types.
```javascript
const getDriveItemContent = async (client, driveId, itemId, name) => {
    try {
        const fileType = path.extname(name).toLowerCase();
        // the below files types are the ones that are able to be converted to PDF to extract the text. See https://learn.microsoft.com/en-us/graph/api/driveitem-get-content-format?view=graph-rest-1.0&tabs=http
        const allowedFileTypes = ['.pdf', '.doc', '.docx', '.odp', '.ods', '.odt', '.pot', '.potm', '.potx', '.pps', '.ppsx', '.ppsxm', '.ppt', '.pptm', '.pptx', '.rtf'];
        // filePath changes based on file type, adding ?format=pdf to convert non-pdf types to pdf for text extraction, so all files in allowedFileTypes above are converted to pdf
        const filePath = `/drives/${driveId}/items/${itemId}/content` + ((fileType === '.pdf' || fileType === '.txt' || fileType === '.csv') ? '' : '?format=pdf');
        if (allowedFileTypes.includes(fileType)) {
            response = await client.api(filePath).getStream();
            // The below takes the chunks in response and combines
            let chunks = [];
            for await (let chunk of response) {
                chunks.push(chunk);
            }
            let buffer = Buffer.concat(chunks);
            // the below extracts the text from the PDF.
            const pdfContents = await pdfParse(buffer);
            return pdfContents.text;
        } else if (fileType === '.txt') {
            // If the type is txt, it does not need to create a stream and instead just grabs the content
            response = await client.api(filePath).get();
            return response;
        }  else if (fileType === '.csv') {
            response = await client.api(filePath).getStream();
            let chunks = [];
            for await (let chunk of response) {
                chunks.push(chunk);
            }
            let buffer = Buffer.concat(chunks);
            let dataString = buffer.toString('utf-8');
            return dataString
            
    } else {
        return 'Unsupported File Type';
    }
     
    } catch (error) {
        console.error('Error fetching drive content:', error);
        throw new Error(`Failed to fetch content for ${name}: ${error.message}`);
    }
};
```

#### Integrating GPT 4o mini for Text Analysis

This function utilizes the OpenAI SDK to analyze text extracted from documents and find relevant information based on a user query. This helps to ensure only relevant text to the user’s question is returned to the GPT. 

```javascript
const getRelevantParts = async (text, query) => {
    try {
        // We use your OpenAI key to initialize the OpenAI client
        const openAIKey = process.env["OPENAI_API_KEY"];
        const openai = new OpenAI({
            apiKey: openAIKey,
        });
        const response = await openai.chat.completions.create({
            // Using gpt-4o-mini due to speed to prevent timeouts. You can tweak this prompt as needed
            model: "gpt-4o-mini",
            messages: [
                {"role": "system", "content": "You are a helpful assistant that finds relevant content in text based on a query. You only return the relevant sentences, and you return a maximum of 10 sentences"},
                {"role": "user", "content": `Based on this question: **"${query}"**, get the relevant parts from the following text:*****\n\n${text}*****. If you cannot answer the question based on the text, respond with 'No information provided'`}
            ],
            // using temperature of 0 since we want to just extract the relevant content
            temperature: 0,
            // using max_tokens of 1000, but you can customize this based on the number of documents you are searching. 
            max_tokens: 1000
        });
        return response.choices[0].message.content;
    } catch (error) {
        console.error('Error with OpenAI:', error);
        return 'Error processing text with OpenAI' + error;
    }
};
```

#### Creating the Azure Function to Handle Requests

Now that we have all these helper functions, the Azure Function will orchestrate the flow, by authenticating the user, performing the search, and iterating through the search results to extract the text and retrieve the relevant parts of the text to the GPT.

**Handling HTTP Requests:** The function starts by extracting the query and searchTerm from the HTTP request. It checks if the Authorization header is present and extracts the bearer token.

**Authentication:** Using the bearer token, it obtains an OBO token from Microsoft's identity platform using getOboToken defined above.

**Initializing the Graph Client:** With the OBO token, it initializes the Microsoft Graph client using initGraphClient defined above.

**Document Search:** It constructs a search query and sends it to the Microsoft Graph API to find documents based on the searchTerm.

**Document Processing**: For each document returned by the search:

- It retrieves the document content using getDriveItemContent.

- If the file type is supported, it analyzes the content using getRelevantParts, which sends the text to OpenAI's model for extracting relevant information based on the query.

- It collects the analysis results and includes metadata like the document name and URL.

**Response**: The function sorts the results by relevance and sends them back in the HTTP response.

```javascript
module.exports = async function (context, req) {
    const query = req.query.query || (req.body && req.body.query);
    const searchTerm = req.query.searchTerm || (req.body && req.body.searchTerm);
    if (!req.headers.authorization) {
        context.res = {
            status: 400,
            body: 'Authorization header is missing'
        };
        return;
    }
    /// The below takes the token passed to the function, to use to get an OBO token.
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
    // Initialize the Graph Client using the initGraphClient function defined above
    let client = initGraphClient(accessToken);
    // this is the search body to be used in the Microsft Graph Search API: https://learn.microsoft.com/en-us/graph/search-concept-files
    const requestBody = {
        requests: [
            {
                entityTypes: ['driveItem'],
                query: {
                    queryString: searchTerm
                },
                from: 0,
                // the below is set to summarize the top 10 search results from the Graph API, but can configure based on your documents. 
                size: 10
            }
        ]
    };

    try { 
        // Function to tokenize content (e.g., based on words). 
        const tokenizeContent = (content) => {
            return content.split(/\s+/);
        };

        // Function to break tokens into 10k token windows for gpt-4o-mini
        const breakIntoTokenWindows = (tokens) => {
            const tokenWindows = []
            const maxWindowTokens = 10000; // 10k tokens
            let startIndex = 0;

            while (startIndex < tokens.length) {
                const window = tokens.slice(startIndex, startIndex + maxWindowTokens);
                tokenWindows.push(window);
                startIndex += maxWindowTokens;
            }

            return tokenWindows;
        };
        // This is where we are doing the search
        const list = await client.api('/search/query').post(requestBody);

        const processList = async () => {
            // This will go through and for each search response, grab the contents of the file and summarize with gpt-4o-mini
            const results = [];

            await Promise.all(list.value[0].hitsContainers.map(async (container) => {
                for (const hit of container.hits) {
                    if (hit.resource["@odata.type"] === "#microsoft.graph.driveItem") {
                        const { name, id } = hit.resource;
                        // We use the below to grab the URL of the file to include in the response
                        const webUrl = hit.resource.webUrl.replace(/\s/g, "%20");
                        // The Microsoft Graph API ranks the reponses, so we use this to order it
                        const rank = hit.rank;
                        // The below is where the file lives
                        const driveId = hit.resource.parentReference.driveId;
                        const contents = await getDriveItemContent(client, driveId, id, name);
                        if (contents !== 'Unsupported File Type') {
                            // Tokenize content using function defined previously
                            const tokens = tokenizeContent(contents);

                            // Break tokens into 10k token windows
                            const tokenWindows = breakIntoTokenWindows(tokens);

                            // Process each token window and combine results
                            const relevantPartsPromises = tokenWindows.map(window => getRelevantParts(window.join(' '), query));
                            const relevantParts = await Promise.all(relevantPartsPromises);
                            const combinedResults = relevantParts.join('\n'); // Combine results

                            results.push({ name, webUrl, rank, contents: combinedResults });
                        } 
                        else {
                            results.push({ name, webUrl, rank, contents: 'Unsupported File Type' });
                        }
                    }
                }
            }));

            return results;
       