# Building a Gmail GPT Action: A Step-by-Step Guide

## Introduction

This guide walks you through creating a custom GPT Action that integrates with Google Gmail. By the end, you'll have a GPT that can read, send, draft, and manage emails on your behalf, acting as a powerful email productivity assistant.

### Prerequisites

Before you begin, ensure you have:
1. A Google Cloud account with billing enabled
2. Access to the [Google Cloud Console](https://console.cloud.google.com/)
3. A ChatGPT Plus or Enterprise subscription (for creating Custom GPTs)

## Step 1: Set Up Google Cloud Project and Enable Gmail API

First, you need to configure the Google Cloud side of the integration.

1. **Create or Select a Google Cloud Project**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Click the project dropdown in the top navigation bar
   - Click "New Project" or select an existing project

2. **Enable the Gmail API**
   - In the left sidebar, navigate to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click on the Gmail API result
   - Click the "Enable" button

3. **Configure OAuth Consent Screen**
   - Navigate to "APIs & Services" > "OAuth consent screen"
   - Select "External" user type (for testing) or "Internal" (for workspace accounts)
   - Fill in the required app information (app name, user support email, developer contact email)
   - Add the scope: `https://www.googleapis.com/auth/gmail.modify`
   - Add test users if using "External" user type (add your own email address)

## Step 2: Create OAuth 2.0 Credentials

Now you'll create the credentials that ChatGPT will use to authenticate with Gmail.

1. **Create OAuth Client ID**
   - Navigate to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Select "Web application" as the application type
   - Give your client a name (e.g., "ChatGPT Gmail Action")

2. **Configure Authorized Redirect URIs**
   - Leave the "Authorized redirect URIs" field empty for now
   - You'll add this later after getting the callback URL from ChatGPT
   - Click "Create"
   - A popup will appear with your **Client ID** and **Client Secret**
   - **Important**: Copy both values and store them securely. You won't be able to see the Client Secret again.

## Step 3: Create Your Custom GPT

Now let's move to the ChatGPT interface to create the GPT Action.

1. **Create a New GPT**
   - Go to [ChatGPT](https://chat.openai.com/)
   - Click your name in the bottom left > "My GPTs"
   - Click "Create a GPT"

2. **Configure the GPT Instructions**
   - In the "Configure" tab, find the "Instructions" section
   - Copy and paste the following instructions:

```markdown
**Context**
Act as an email assistant designed to enhance user interaction with emails in various ways. This GPT can assist with productivity by summarizing emails/threads, identifying next steps/follow-ups, drafting or sending pre-written responses, and programmatically interacting with third-party tools (e.g., Notion to-dos, Slack channel summaries, data extraction for responses). This GPT has full scope access to the GMAIL OAuth 2.0 API, capable of reading, composing, sending, and permanently deleting emails from Gmail.

**Instructions**
- Always conclude an email by signing off with logged in user's name, unless otherwise stated.
- Verify that the email data is correctly encoded in the required format (e.g., base64 for the message body).
- Email Encoding Process: 1) Construct the email message in RFC 2822 format. 2) Base64 encode the email message. 3) Send the encoded message using the API.
- If not specified, sign all emails with the user name.
- API Usage: After answering the user's question, do not call the Google API again until another question is asked.
- All emails created, draft or sent, should be in plain text.
- Ensure that the email format is clean and is formatted as if someone sent the email from their own inbox. Once a draft is created or email sent, display a message to the user confirming that the draft is ready or the email is sent.
- Check that the "to" email address is valid and in the correct format. It should be in the format "recipient@example.com".
- Only provide summaries of existing emails; do not fabricate email content.
- Professionalism: Behave professionally, providing clear and concise responses.
- Clarification: Ask for clarification when needed to ensure accuracy and completeness in fulfilling user requests.
- Privacy and Security: Respect user privacy and handle all data securely.
```

## Step 4: Add the OpenAPI Schema

1. **Navigate to the Actions Section**
   - In the GPT configuration, click "Add Action"
   - Select "Import from URL" or choose to paste the schema directly

2. **Add the OpenAPI Schema**
   - Copy and paste the following OpenAPI schema into the schema field:

```yaml
openapi: 3.1.0

info:
  title: Gmail Email API
  version: 1.0.0
  description: API to read, write, and send emails in a Gmail account.

servers:
  - url: https://gmail.googleapis.com

paths:
  /gmail/v1/users/{userId}/messages:
    get:
      summary: List All Emails
      description: Lists all the emails in the user's mailbox.
      operationId: listAllEmails
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
        - name: q
          in: query
          schema:
            type: string
          description: Query string to filter messages (optional).
        - name: pageToken
          in: query
          schema:
            type: string
          description: Token to retrieve a specific page of results in the list.
        - name: maxResults
          in: query
          schema:
            type: integer
            format: int32
          description: Maximum number of messages to return.
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MessageList'
        '400':
          description: Bad Request
        '401':
          description: Unauthorized
        '403':
          description: Forbidden
        '404':
          description: Not Found
        '500':
          description: Internal Server Error

  /gmail/v1/users/{userId}/messages/send:
    post:
      summary: Send Email
      description: Sends a new email.
      operationId: sendEmail
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Message'
      responses:
        '200':
          description: Email sent successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Message'
        '400':
          description: Bad Request
        '401':
          description: Unauthorized
        '403':
          description: Forbidden
        '500':
          description: Internal Server Error

  /gmail/v1/users/{userId}/messages/{id}:
    get:
      summary: Read Email
      description: Gets the full email content including headers and body.
      operationId: readEmail
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
        - name: id
          in: path
          required: true
          schema:
            type: string
          description: The ID of the email to retrieve.
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FullMessage'
        '400':
          description: Bad Request
        '401':
          description: Unauthorized
        '403':
          description: Forbidden
        '404':
          description: Not Found
        '500':
          description: Internal Server Error

  /gmail/v1/users/{userId}/messages/{id}/modify:
    post:
      summary: Modify Label
      description: Modify labels of an email.
      operationId: modifyLabels
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
        - name: id
          in: path
          required: true
          schema:
            type: string
          description: The ID of the email to change labels.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/LabelModification'
      responses:
        '200':
          description: Labels modified successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Message'
        '400':
          description: Bad Request
        '401':
          description: Unauthorized
        '403':
          description: Forbidden
        '500':
          description: Internal Server Error

  /gmail/v1/users/{userId}/drafts:
    post:
      summary: Create Draft
      description: Creates a new email draft.
      operationId: createDraft
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Draft'
      responses:
        '200':
          description: Draft created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Draft'
        '400':
          description: Bad Request
        '401':
          description: Unauthorized
        '403':
          description: Forbidden
        '500':
          description: Internal Server Error

  /gmail/v1/users/{userId}/drafts/send:
    post:
      summary: Send Draft
      description: Sends an existing email draft.
      operationId: sendDraft
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SendDraftRequest'
      responses:
        '200':
          description: Draft sent successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Message'
        '400':
          description: Bad Request
        '401':
          description: Unauthorized
        '403':
          description: Forbidden
        '500':
          description: Internal Server Error

components:
  schemas:
    MessageList:
      type: object
      properties:
        messages:
          type: array
          items:
            $ref: '#/components/schemas/Message'
        nextPageToken:
          type: string

    Message:
      type: object
      properties:
        id:
          type: string
        threadId:
          type: string
        labelIds:
          type: array
          items:
            type: string
        addLabelIds:
          type: array
          items:
            type: string
        removeLabelIds:
          type: array
          items:
            type: string
        snippet:
          type: string
        raw:
          type: string
          format: byte
          description: The entire email message in an RFC 2822 formatted and base64url encoded string.

    FullMessage:
      type: object
      properties:
        id:
          type: string
        threadId:
          type: string
        labelIds:
          type: array
          items:
            type: string
        snippet:
          type: string
        payload:
          type: object
          properties:
            headers:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  value:
                    type: string
            parts:
              type: array
              items:
                type: object
                properties:
                  mimeType:
                    type: string
                  body:
                    type: object
                    properties:
                      data:
                        type: string

    LabelModification:
      type: object
      properties:
        addLabelIds:
          type: array
          items:
            type: string
        removeLabelIds:
          type: array
          items:
            type: string

    Label:
      type: object
      properties:
        addLabelIds:
          type: array
          items:
            type: string
        removeLabelIds:
          type: array
          items:
            type: string

    EmailDraft:
      type: object
      properties:
        to:
          type: array
          items:
            type: string
        cc:
          type: array
          items:
            type: string
        bcc:
          type: array
          items:
            type: string
        subject:
          type: string
        body:
          type: object
          properties:
            mimeType:
              type: string
              enum: [text/plain, text/html]
            content:
              type: string

    Draft:
      type: object
      properties:
        id:
          type: string
        message:
          $ref: '#/components/schemas/Message'

    SendDraftRequest:
      type: object
      properties:
        draftId:
          type: string
          description: The ID of the draft to send.
        userId:
          type: string
          description: The user's email address. Use "me" to indicate the authenticated user.
```

## Step 5: Configure Authentication

1. **Get the Callback URL from ChatGPT**
   - In the GPT configuration, click "Authentication"
   - Select "OAuth"
   - A "Callback URL" will be generated
   - Copy this URL

2. **Complete Google Cloud OAuth Configuration**
   - Return to the Google Cloud Console
   - Navigate to "APIs & Services" > "Credentials"
   - Click on your OAuth 2.0 Client ID
   - Click "Edit OAuth Client"
   - In the "Authorized redirect URIs" section, click "Add URI"
   - Paste the callback URL you copied from ChatGPT
   - Click "Save"

3. **Configure OAuth in ChatGPT**
   - Back in ChatGPT GPT configuration, fill in the authentication fields:
     - **Client ID**: Paste the Client ID from Google Cloud
     - **Client Secret**: Paste the Client Secret from Google Cloud
     - **Authorization URL**: `https://accounts.google.com/o/oauth2/auth`
     - **Token URL**: `https://oauth2.googleapis.com/token`
     - **Scope**: `https://mail.google.com/`
     - **Token**: Default (POST)

## Step 6: Test Your Gmail GPT Action

1. **Save and Test**
   - Click "Save" in the GPT configuration
   - Select "Only me" or "Anyone with a link" depending on your sharing preference
   - Click "Confirm"

2. **Initial Authentication**
   - Open your new GPT
   - The first time you use it, you'll be prompted to authenticate with Google
   - Follow the OAuth flow to grant permissions to your Gmail account

3. **Test Basic Functions**
   Try these commands to test your GPT Action:
   - "List my recent emails"
   - "Send an email to [test@example.com] with subject 'Test' and body 'This is a test email'"
   - "Create a draft email about our meeting tomorrow"

## Troubleshooting Common Issues

### Callback URL Error
If you encounter a callback URL error in ChatGPT:
- Double-check that you've added the exact callback URL from ChatGPT to your Google Cloud OAuth client's "Authorized redirect URIs"
- Ensure there are no trailing spaces or characters
- The URL should look like: `https://chat.openai.com/aip/[unique-id]/oauth/callback`

### Authentication Failures
If authentication fails:
- Verify your Client ID and Client Secret are correct
- Ensure the Gmail API is enabled in your Google Cloud project
- Check that you've added your email as a test user in the OAuth consent screen (if using "External" user type)

### API Permission Issues
If the GPT can't access certain functions:
- Verify the scope `https://mail.google.com/` is correctly set in both Google Cloud and ChatGPT
- Check that you granted all requested permissions during the OAuth flow

## Next Steps and Advanced Configuration

Once your basic Gmail GPT Action is working, consider these enhancements:

1. **Add Additional Scopes**: Depending on your needs, you might add more Gmail scopes
2. **Integrate with Other Services**: Combine with other GPT Actions for workflows like:
   - Saving email summaries to Notion
   - Creating tasks from email follow-ups
   - Analyzing email sentiment over time
3. **Custom Instructions**: Refine your GPT instructions based on your specific use cases

## Conclusion

You've successfully created a Gmail GPT Action that can manage your email communications. This powerful tool can now help you with email summarization, drafting responses, and managing your inbox more efficiently. Remember to always review sensitive emails before sending and ensure the GPT's actions align with your communication standards.