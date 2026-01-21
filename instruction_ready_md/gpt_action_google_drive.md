# Building a GPT Action for Google Drive: A Developer's Guide

## Introduction

This guide walks you through building a GPT Action that connects ChatGPT to Google Drive, enabling natural language interactions with your files. This action allows you to list, search, and read files from Google Drive, making it perfect for working with meeting minutes, design documents, memos, and other smaller files.

### Prerequisites

Before starting, ensure you have:

1. A Google Cloud account
2. A Google Cloud project with the Drive API enabled
3. Basic familiarity with [GPT Actions](https://platform.openai.com/docs/actions/introduction)

## Step 1: Set Up Your Google Cloud Project

First, configure your Google Cloud environment to enable the Drive API and create OAuth credentials.

1. **Enable the Drive API:**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to "APIs & Services" → "Library"
   - Search for "Google Drive API" and enable it

2. **Configure OAuth Consent Screen:**
   - Go to "APIs & Services" → "OAuth consent screen"
   - Set up the consent screen (external or internal)
   - Add test users if using "Testing" publishing status

3. **Create OAuth Credentials:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Select "Web application" as the application type
   - Note your **Client ID** and **Client Secret** for later use

## Step 2: Configure Authentication in ChatGPT

Now, set up OAuth authentication in your Custom GPT.

1. In your Custom GPT editor, navigate to the "Configure" tab
2. Click on "Authentication" and select **"OAuth"**
3. Enter the following authentication details:

```
Client ID: [Your Google Cloud Client ID]
Client Secret: [Your Google Cloud Client Secret]
Authorization URL: https://accounts.google.com/o/oauth2/auth
Token URL: https://oauth2.googleapis.com/token
Scope: https://www.googleapis.com/auth/drive.readonly
Token: Default (POST)
Privacy Policy: https://policies.google.com/privacy?hl=en-US
```

4. **Important:** Copy the callback URL provided by ChatGPT
5. Return to your Google Cloud Console and add this callback URL to your OAuth credentials under "Authorized redirect URIs"

## Step 3: Define Your Custom GPT Instructions

Configure your GPT's behavior by adding specific instructions. These guide how the GPT interacts with your Google Drive files.

```markdown
*** Context ***

You are an office helper who takes a look at files within Google Drive and reads in information. When asked about something, please take a look at all of the relevant information within the drive. Respect file names, but also take a look at each document and sheet.

*** Instructions ***

Use the 'listFiles' function to get a list of files available within docs. Determine which files make the most sense to pull back, taking into account name and title. After the output of listFiles is called into context, act like a normal business analyst.

Things you should do:
- **Summaries:** Read through entire files before providing consistent, concise summaries
- **Professionalism:** Provide clear and concise responses
- **Synthesis, Coding, and Data Analysis:** Explain coding blocks when used
- **Date Handling:** Search using date fields; if nothing found, use titles
- **Clarification:** Ask for clarification when needed to ensure accuracy
- **Privacy and Security:** Respect user privacy and handle all data securely
```

## Step 4: Implement the OpenAPI Schema

Add the following OpenAPI schema to define your GPT's available actions. This schema provides three key functions: listing files, getting metadata, and exporting files.

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Google Drive API",
    "description": "API for interacting with Google Drive",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "https://www.googleapis.com/drive/v3"
    }
  ],
  "paths": {
    "/files": {
      "get": {
        "operationId": "ListFiles",
        "summary": "List files",
        "description": "Retrieve a list of files in the user's Google Drive.",
        "parameters": [
          {
            "name": "q",
            "in": "query",
            "description": "Query string for searching files.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "includeItemsFromAllDrives",
            "in": "query",
            "description": "Whether both My Drive and shared drive items should be included in results.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "supportsAllDrives",
            "in": "query",
            "description": "Whether the requesting application supports both My Drives and shared drives.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "pageSize",
            "in": "query",
            "description": "Maximum number of files to return.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 10
            }
          },
          {
            "name": "pageToken",
            "in": "query",
            "description": "Token for continuing a previous list request.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "fields",
            "in": "query",
            "description": "Comma-separated list of fields to include in the response.",
            "required": false,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A list of files.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "kind": {
                      "type": "string",
                      "example": "drive#fileList"
                    },
                    "nextPageToken": {
                      "type": "string",
                      "description": "Token to retrieve the next page of results."
                    },
                    "files": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "properties": {
                          "id": {
                            "type": "string"
                          },
                          "name": {
                            "type": "string"
                          },
                          "mimeType": {
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
    },
    "/files/{fileId}": {
      "get": {
        "operationId": "getMetadata",
        "summary": "Get file metadata",
        "description": "Retrieve metadata for a specific file.",
        "parameters": [
          {
            "name": "fileId",
            "in": "path",
            "description": "ID of the file to retrieve.",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "fields",
            "in": "query",
            "description": "Comma-separated list of fields to include in the response.",
            "required": false,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Metadata of the file.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "id": {
                      "type": "string"
                    },
                    "name": {
                      "type": "string"
                    },
                    "mimeType": {
                      "type": "string"
                    },
                    "description": {
                      "type": "string"
                    },
                    "createdTime": {
                      "type": "string",
                      "format": "date-time"
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/files/{fileId}/export": {
      "get": {
        "operationId": "export",
        "summary": "Export a file",
        "description": "Export a Google Doc to the requested MIME type.",
        "parameters": [
          {
            "name": "fileId",
            "in": "path",
            "description": "ID of the file to export.",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "mimeType",
            "in": "query",
            "description": "The MIME type of the format to export to.",
            "required": true,
            "schema": {
              "type": "string",
              "enum": [
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain"
              ]
            }
          }
        ],
        "responses": {
          "200": {
            "description": "The exported file.",
            "content": {
              "application/pdf": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              },
              "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              },
              "text/plain": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              }
            }
          },
          "400": {
            "description": "Invalid MIME type or file ID."
          },
          "404": {
            "description": "File not found."
          }
        }
      }
    }
  }
}
```

## Step 5: Test Your GPT Action

Now that your GPT Action is configured, test it with various queries:

1. **List files:** "Show me all files related to Q4 planning"
2. **Search files:** "Find documents containing 'budget' in the title"
3. **Read content:** "What does the marketing strategy document say about social media?"
4. **Summarize:** "Give me a summary of the project status report"

### Query Examples for the `listFiles` Function

Here are useful query patterns for searching Google Drive:

| What you want to query | Example Query |
|------------------------|---------------|
| Files with the name "hello" | `name = 'hello'` |
| Files with a name containing "hello" and "goodbye" | `name contains 'hello' and name contains 'goodbye'` |
| Files that are not folders | `mimeType != 'application/vnd.google-apps.folder'` |
| Files that contain the text "important" | `fullText contains 'important'` |
| Files modified after a given date | `modifiedTime > '2024-01-01T00:00:00'` |
| Files shared with the authorized user | `sharedWithMe and name contains 'hello'` |

## Troubleshooting

### Common Issues and Solutions

1. **Callback URL Error:**
   - Ensure you've added the exact callback URL from ChatGPT to your Google Cloud OAuth credentials
   - Check for typos or extra spaces in the URL

2. **Authentication Failures:**
   - Verify your Client ID and Client Secret are correct
   - Ensure the Drive API is enabled in your Google Cloud project
   - Check that test users are added if using "Testing" publishing status

3. **Permission Errors:**
   - The `drive.readonly` scope provides read-only access
   - For write access, use `https://www.googleapis.com/auth/drive` instead

## Best Practices

1. **Use Export Instead of Get:** When reading file content, prefer the `export` function over `get` to avoid downloading entire files unnecessarily.

2. **Optimize Queries:** Use specific query parameters to filter results and reduce API calls.

3. **Handle Large Files:** This action works best with smaller files. For large documents or complex spreadsheets, consider building specialized GPTs for Google Docs or Google Sheets.

4. **Test Thoroughly:** Test your GPT with various file types and queries to ensure it handles edge cases properly.

## Next Steps

Your Google Drive GPT Action is now ready! Users can interact with their Drive files using natural language, making it easier to find information, summarize documents, and answer questions based on file content.

For more advanced functionality, consider extending this action with additional Google Drive API methods or integrating it with other Google Workspace applications.