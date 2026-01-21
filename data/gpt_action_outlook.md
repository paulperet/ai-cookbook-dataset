# Guide: Building a GPT Action for Microsoft Outlook

This guide walks you through creating a Custom GPT Action that connects to Microsoft Outlook. Your GPT will be able to send emails, read messages, and manage calendar events using the Microsoft Graph API.

## Prerequisites

Before you begin, ensure you have:
1.  Access to the [Azure Portal](https://portal.azure.com) with permissions to create App Registrations.
2.  A ChatGPT Plus or Enterprise subscription to create Custom GPTs.

## Step 1: Register Your Application in Azure

First, you need to create an App Registration in Azure to act as the secure bridge between your GPT and Outlook.

1.  Navigate to the [Azure Portal](https://portal.azure.com).
2.  Go to **Azure Active Directory** > **App registrations** and click **New registration**.
3.  Give your application a name (e.g., "My GPT Outlook Action").
4.  For **Supported account types**, select **Accounts in any organizational directory (Any Azure AD directory - Multitenant)**.
5.  Leave the **Redirect URI** section blank for now. You will add this later.
6.  Click **Register**.

## Step 2: Create a Client Secret

Your GPT needs a secure key to authenticate with Azure.

1.  In your new App Registration, navigate to **Certificates & secrets** in the sidebar.
2.  Click **New client secret**.
3.  Provide a description and choose an expiry period.
4.  Click **Add**.
5.  **Crucially, copy the `Value` of the new secret immediately.** Store it securely, as you will not be able to see it again.

## Step 3: Configure API Permissions

Define what your GPT is allowed to do with the user's Outlook data.

1.  In your App Registration, go to **API permissions**.
2.  Click **Add a permission**, then select **Microsoft Graph** > **Delegated permissions**.
3.  Using the search bar, find and add the following permissions:
    *   `Calendars.ReadWrite`
    *   `Mail.Read`
    *   `Mail.Send`
    *   `User.Read`
4.  Click **Add permissions**.
5.  **Important:** Click **Grant admin consent for [Your Directory Name]** to activate these permissions.

## Step 4: Configure Your Custom GPT

Now, you will create the GPT and configure its connection to your Azure app.

### 4.1 Set the GPT's Instructions

1.  In ChatGPT, navigate to **Create a GPT**.
2.  In the **Configure** tab, find the **Instructions** field.
3.  Paste the following instructions to define your GPT's behavior:

```
**Context**: You are a specialized GPT designed to manage emails and calendar events through API connections to Microsoft Outlook. This GPT can create, read, send, and alter emails and calendar events based on user instructions. It ensures efficient handling of communication and scheduling needs by leveraging Microsoft Graph API for seamless integration with Outlook services.

**Instructions**:
- When asked to perform a task, use the available actions via the microsoft.graph.com API.
- You should behave professionally and provide clear, concise responses.
- Offer assistance with tasks such as drafting emails, scheduling meetings, organising calendar events, and retrieving email or event details.
- Ask for clarification when needed to ensure accuracy and completeness in fulfilling user requests.
- Always conclude an email by signing off with the logged-in user's name, which can be retrieved via the User.Read endpoint.
```

### 4.2 Add the OpenAPI Schema (Actions)

1.  In the same **Configure** tab, scroll to the **Actions** section and click **Create new action**.
2.  In the **Schema** field, paste the following OpenAPI specification. This defines the specific API endpoints your GPT can call.

```yaml
openapi: 3.1.0
info:
  title: Microsoft Graph API Integration
  version: 1.0.0
servers:
  - url: https://graph.microsoft.com/v1.0
paths:
  /me:
    get:
      operationId: getUserProfile
      summary: Get the authenticated user's profile
      responses:
        '200':
          description: A user profile
  /me/messages:
    get:
      operationId: getUserMessages
      summary: Get the authenticated user's messages
      parameters:
        - name: $top
          in: query
          required: false
          schema:
            type: integer
            default: 10
        - name: $filter
          in: query
          required: false
          schema:
            type: string
        - name: $orderby
          in: query
          required: false
          schema:
            type: string
      responses:
        '200':
          description: A list of user messages
  /me/sendMail:
    post:
      operationId: sendUserMail
      summary: Send an email as the authenticated user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
      responses:
        '202':
          description: Accepted
  /me/events:
    get:
      operationId: getUserCalendarEvents
      summary: Get the authenticated user's calendar events
      responses:
        '200':
          description: A list of calendar events
    post:
      operationId: createUserCalendarEvent
      summary: Create a new calendar event for the authenticated user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
      responses:
        '201':
          description: Created
```

### 4.3 Configure OAuth Authentication

This step connects your GPT to the Azure App Registration you created.

1.  In the **Authentication** section within the Action configuration, select **OAuth**.
2.  Fill in the details using the information from your Azure App Registration:
    *   **Client ID**: Found on your App's **Overview** page as **Application (client) ID**.
    *   **Client Secret**: The secret `Value` you saved in Step 2.
    *   **Authorization URL**: `https://login.microsoftonline.com/<Tenant_ID>/oauth2/v2.0/authorize`
        *   Replace `<Tenant_ID>` with the **Directory (tenant) ID** from your App's Overview page.
    *   **Token URL**: `https://login.microsoftonline.com/<Tenant_ID>/oauth2/v2.0/token`
        *   Replace `<Tenant_ID>` as above.
    *   **Scope**: `https://graph.microsoft.com/User.Read https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/Mail.Read https://graph.microsoft.com/Calendars.ReadWrite`
    *   **Token Exchange Method**: Keep as **POST Request**.

## Step 5: Finalize the Azure Configuration

After setting up authentication in ChatGPT, you must complete the configuration loop in Azure.

1.  In your GPT's Action **Authentication** section, find and copy the **Callback URL**.
2.  Go back to your Azure App Registration and navigate to **Authentication**.
3.  Click **Add a platform** and select **Web**.
4.  Paste the copied **Callback URL** into the **Redirect URIs** field.
5.  Click **Configure**.

## Testing and Next Steps

Your GPT Action is now configured. Save and publish your Custom GPT. The first time a user interacts with it, they will be prompted to authenticate with their Microsoft account, granting the permissions you specified.

You can now instruct your GPT with natural language commands like:
*   "Send an email to john@example.com about the project update."
*   "What are my meetings for tomorrow?"
*   "Read my last 5 emails."

## Troubleshooting

*   **Callback URL Error**: If you encounter an authentication error mentioning the callback URL, double-check that the URL pasted in your Azure App's **Redirect URIs** matches the one shown in your GPT's Action settings exactly. This URL can change if you modify the authentication settings.
*   **Insufficient Permissions**: Ensure you clicked **Grant admin consent** in Step 3. Users may need their admin to approve the application if admin consent was not granted.