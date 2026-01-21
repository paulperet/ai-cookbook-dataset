# Building a Custom GPT Action for Box: A Step-by-Step Guide

## Introduction

This guide walks you through creating a custom GPT Action that connects to your Box.com account. This integration enables ChatGPT to search, retrieve, and analyze files and folders stored in Box, providing powerful insights directly within your conversations.

The setup requires two distinct actions:
1.  **Box API Action:** Directly queries the Box API for file metadata, folder structures, and search results.
2.  **Azure Function Action (Middleware):** Formats the Box API's response to allow ChatGPT to download and read the actual *contents* of files. This action runs transparently in the background.

### Prerequisites

Before you begin, ensure you have the following:

1.  **A Box Developer Account:** You need this to create a custom OAuth application.
    *   Sign up at: [https://developer.box.com/](https://developer.box.com/)
2.  **Access to Azure Portal:** You will need permissions to create Azure Function Apps and App Registrations to deploy the middleware.
3.  **OpenAI GPTs Access:** You must have the ability to create and configure custom GPTs.

## Part 1: Setting Up Your Box Application

First, you need to create and configure an application in Box that your GPT will authenticate with.

### Step 1: Create a Box Custom App

1.  Log into your [Box Developer Console](https://developer.box.com/).
2.  Click **"Create New App"**.
3.  Select **"Custom App"** as the app type.
4.  Choose **"OAuth 2.0 with JWT (Server Authentication)"** or the standard **"User Authentication (OAuth 2.0)"** type. This guide assumes OAuth 2.0.
5.  Give your app a name (e.g., "ChatGPT Box Integration") and click **"Create App"**.

### Step 2: Configure OAuth Settings

Navigate to the **Configuration** tab of your newly created app. You will need information from several sections.

1.  **OAuth 2.0 Credentials:** Note down your **Client ID** and **Client Secret**. You will enter these into ChatGPT later.
    *   **Important:** Keep your Client Secret confidential.
2.  **Application Scopes:** Your GPT needs specific permissions to function. Under the **Application Scopes** section, select at least the following:
    *   `Read all files and folders stored in Box`
    *   `Manage Enterprise properties` (often required for metadata operations)
    *   Add any other `read` scopes relevant to the endpoints you plan to use (e.g., `Read tasks`, `Read webhooks`).
3.  **Redirect URIs:** **Leave this field blank for now.** You will return here to add the Redirect URI *after* you create the Action in the ChatGPT interface in a later step.

Keep your Box Developer Console open, as you will need to return to the **Redirect URIs** field.

## Part 2: Configuring Your Custom GPT

Now, you will create the GPT and configure its instructions and first action.

### Step 3: Create a New GPT and Add Instructions

1.  In the ChatGPT interface, navigate to **"Explore GPTs"** and click **"Create a GPT"**.
2.  In the **Configure** tab, find the **Instructions** field.
3.  Copy and paste the following instructions to guide your GPT's behavior:

```
**Context**
This GPT connects to your Box.com account to search files and folders, providing accurate and helpful responses based on your queries. It assists with finding, organizing, and retrieving information stored in Box.com. It ensures secure and private handling of any accessed data and will not modify or delete files unless explicitly instructed.

Please use the Box API documentation for reference: https://developer.box.com/reference/

Users can search with the Box search endpoint or Box metadata search endpoint.

**Instructions**
- When retrieving file information from Box, provide as much detail as possible.
- Format results into a table when more than one file is returned. Include headers like Name, ID, Modified Date, Created Date, Size, and any other valuable metadata.
- Provide insights on files and suggest patterns for users. Give example queries and suggestions when appropriate.
- When a user wants to compare files, retrieve the files for the user without asking for confirmation.
- Ask for clarification if a request is ambiguous or if additional details are needed to perform a search.
- Maintain a professional and friendly tone.
```

### Step 4: Create Action 1 - The Box API Action

In the same **Configure** tab, navigate to the **Actions** section and click **"Create new action"**.

1.  In the **Schema** input field, you will paste an OpenAPI specification. This defines how the GPT can interact with the Box API.
2.  Copy the entire JSON schema provided below and paste it into the Schema field.

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Box.com API",
    "description": "API for Box.com services",
    "version": "v1.0.0"
  },
  "servers": [
    {
      "url": "https://api.box.com/2.0"
    }
  ],
  "paths": {
    "/folders/{folder_id}": {
      "get": {
        "summary": "Get Folder Items",
        "operationId": "getFolderItems",
        "parameters": [
          {
            "name": "folder_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "The ID of the folder"
          }
        ],
        "responses": {
          "200": {
            "description": "A list of items in the folder",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FolderItems"
                }
              }
            }
          }
        },
        "security": [
          {
            "OAuth2": [
              "read:folders"
            ]
          }
        ]
      }
    },
    "/files/{file_id}": {
      "get": {
        "summary": "Get File Information",
        "operationId": "getFileInfo",
        "parameters": [
          {
            "name": "file_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "The ID of the file"
          }
        ],
        "responses": {
          "200": {
            "description": "File information",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FileInfo"
                }
              }
            }
          }
        },
        "security": [
          {
            "OAuth2": [
              "read:files"
            ]
          }
        ]
      }
    },
    "/search": {
      "get": {
        "summary": "Search",
        "operationId": "search",
        "parameters": [
          {
            "name": "query",
            "in": "query",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "Search query"
          }
        ],
        "responses": {
          "200": {
            "description": "Search results",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SearchResults"
                }
              }
            }
          }
        },
        "security": [
          {
            "OAuth2": [
              "search:items"
            ]
          }
        ]
      }
    }
  },
  "components": {
    "schemas": {
      "FolderItems": {
        "type": "object",
        "properties": {
          "total_count": {
            "type": "integer",
            "description": "The total number of items in the folder"
          },
          "entries": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "type": {
                  "type": "string",
                  "description": "The type of the item (e.g., file, folder)"
                },
                "id": {
                  "type": "string",
                  "description": "The ID of the item"
                },
                "name": {
                  "type": "string",
                  "description": "The name of the item"
                }
              }
            }
          }
        }
      },
      "FileInfo": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "The ID of the file"
          },
          "name": {
            "type": "string",
            "description": "The name of the file"
          },
          "size": {
            "type": "integer",
            "description": "The size of the file in bytes"
          },
          "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "The creation time of the file"
          },
          "modified_at": {
            "type": "string",
            "format": "date-time",
            "description": "The last modification time of the file"
          }
        }
      },
      "SearchResults": {
        "type": "object",
        "properties": {
          "total_count": {
            "type": "integer",
            "description": "The total number of search results"
          },
          "entries": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "type": {
                  "type": "string",
                  "description": "The type of the item (e.g., file, folder)"
                },
                "id": {
                  "type": "string",
                  "description": "The ID of the item"
                },
                "name": {
                  "type": "string",
                  "description": "The name of the item"
                }
              }
            }
          }
        }
      }
    },
    "securitySchemes": {
      "OAuth2": {
        "type": "oauth2",
        "flows": {
          "authorizationCode": {
            "authorizationUrl": "https://account.box.com/api/oauth2/authorize",
            "tokenUrl": "https://api.box.com/oauth2/token",
            "scopes": {
              "read:folders": "Read folders",
              "read:files": "Read files",
              "search:items": "Search items"
            }
          }
        }
      }
    }
  }
}
```

**Note:** This schema includes core endpoints for folders, files, and search. You can extend it by adding more paths from the [Box API Reference](https://developer.box.com/reference/), such as `/events`, `/metadata_templates`, etc., following the same pattern.

### Step 5: Configure Authentication for Action 1

With the schema pasted, you now need to set up the OAuth connection so your GPT can log into Box.

1.  In the Action configuration, click on the **"Authentication"** dropdown and select **"OAuth"**.
2.  Fill in the OAuth configuration fields with the details from your Box app:
    *   **Client ID:** Paste the **Client ID** from your Box app's Configuration tab.
    *   **Client Secret:** Paste the **Client Secret** from your Box app.
    *   **Authorization URL:** Use the standard Box OAuth URL:  
        `https://account.box.com/api/oauth2/authorize`
    *   **Token URL:** Use the standard Box token URL:  
        `https://api.box.com/oauth2/token`
3.  **Critical Step - Retrieve the Redirect URI:** After filling in the above fields, click **"Save"** in the GPT editor. ChatGPT will generate a unique Redirect URI for this action. **Copy this generated Redirect URI.**

### Step 6: Complete the Box App Configuration

Now, return to your Box Developer Console.

1.  Go back to your app's **Configuration** tab.
2.  In the **Redirect URIs** field under OAuth 2.0 Settings, paste the Redirect URI you just copied from ChatGPT.
3.  Click **"Save Changes"** to update your Box application.

Your first action is now configured. Users will be able to authenticate their Box account, and the GPT will be able to query metadata about files and folders.

## Part 3: Implementing the Middleware (Action 2)

The Box API Action (Action 1) retrieves metadata. To enable ChatGPT to read the actual *content* of a file (e.g., the text in a PDF or DOCX), you need a middleware function. This is because the Box API returns a web link to download the file, which ChatGPT cannot process directly. The middleware downloads the file, extracts its text content, and returns it in a format ChatGPT can understand.

### Step 7: Deploy the Azure Function

This step involves creating an Azure Function that acts as the middleware. The core logic of this function is to:
1.  Accept a file ID from the GPT.
2.  Use a Box SDK with a valid access token to get a direct download URL for the file.
3.  Download the file content.
4.  Use a library (like `PyPDF2` for PDFs, `python-docx` for Word docs, or `txt` for plain text) to extract the text.
5.  Return the extracted text in a simple JSON response.

**Prerequisite:** You must have an Azure account and the Azure CLI or portal access to create a Function App.

Here is a conceptual Python code snippet for the Azure Function (`__init__.py`):

```python
import logging
import azure.functions as func
from boxsdk import Client, OAuth2
import requests
import PyPDF2
import io
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing request for Box file content.')

    # 1. Get file_id from request
    req_body = req.get_json()
    file_id = req_body.get('file_id')

    if not file_id:
        return func.HttpResponse("Please pass a file_id in the request body", status_code=400)

    # 2. Authenticate with Box (using pre-configured JWT or stored token)
    # This example assumes a Service Account JWT auth for server-to-server.
    # For user context, you would need to store/retrieve the user's OAuth token.
    auth = OAuth2(
        client_id=os.environ["BOX_CLIENT_ID"],
        client_secret=os.environ["BOX_CLIENT_SECRET"],
        access_token=os.environ["BOX_DEVELOPER_TOKEN"] # Or retrieve from a token store
    )
    client = Client(auth)

    try:
        # 3. Get the file
        file_to_download = client.file(file_id).get()
        # 4. Get download URL and download content
        download_url = file_to_download.get_download_url()
        response = requests.get(download_url)
        response.raise_for_status()
        file_content = response.content

        # 5. Extract text based on file type (simplified example for PDF)
        # In production, check file extension or MIME type.
        text = ""
        if file_to_download.name.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        else:
            # Fallback: try to decode as text
            text = file_content.decode('utf-8', errors='ignore')

        # 6. Return the text
        return func.HttpResponse(
            json.dumps({"file_name": file_to_download.name, "content": text}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing file {file_id}: {e}")
        return func.HttpResponse(f"Error retrieving file content: {e}", status_code=500)
```

**Deployment Steps (Overview):**
1.  Create a new Function App in the Azure Portal.
2.  Create a new HTTP Trigger function (Python runtime).
3.  Paste the code above into the function editor.
4.  Add the required Python packages (`boxsdk`, `requests`, `PyPDF2`) to your `requirements.txt`.
5.  Configure the Application Settings for your function to store your `BOX_CLIENT_ID`, `BOX_CLIENT_SECRET`, and `BOX_DEVELOPER_TOKEN` securely.
6.  Deploy the function and note its public URL (e.g., `https://your-function.azurewebsites.net/api/GetBoxFileContent`).

### Step 8: Create Action 2 in Your GPT

Back in your GPT's **Configure** tab, create a second new action.

1.  In the **Schema** field, paste a simple OpenAPI schema that points to your Azure Function. This action is simpler, as its only job is to call your middleware with a file ID.

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Box File Content Fetcher",
    "description": "Middleware to fetch and extract text content from Box files.",
    "version": "v1.0.0"
  },
  "servers": [
    {
      "url": "https://your-function.azurewebsites.net/api" // Replace with your function's base URL
    }
  ],
  "paths": {
    "/GetBoxFileContent": { // Replace with your function's route
      "post": {
        "operationId": "fetchFileContent",
        "summary": "Get the text content of a Box file by its ID.",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "file_id": {
                    "type": "string",
                    "description": "The Box File ID"
                  }
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "File content retrieved successfully",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "file_name": {
                      "type": "string"
                    },
                    "content": {
                      "type": "string"
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

2.  **Important:** This action typically does **not** require separate OAuth configuration if your Azure Function handles its own authentication with Box (e.g., using a service account). Ensure your function's authentication method is secure.

## Conclusion

You have now configured a two-part GPT Action for Box:
*   **Action 1** handles user authentication and queries the