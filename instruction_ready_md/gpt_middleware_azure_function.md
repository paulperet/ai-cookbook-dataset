# Guide: Building a GPT Action Middleware with Azure Functions

## Introduction

This guide walks you through building middleware using an Azure Function to connect a GPT Action to a backend application. This approach allows you to pre-process API responses, handle different data types, and overcome limitations like token constraints.

### Prerequisites

Before you begin, ensure you have:
- An Azure Portal account with permissions to create Function Apps and App Registrations.
- Basic familiarity with [GPT Actions](https://platform.openai.com/docs/actions) and the [Actions Library](https://platform.openai.com/docs/actions/actions-library).

## Step 1: Create an Azure Function App

You will create a Node.js Azure Function App that supports in-portal editing.

1. Navigate to the Azure Portal and create a new **Function App**.
2. Configure the creation with these settings:
    * **Publish**: Code
    * **Runtime stack**: Node.js
    * **Operating system**: Windows
3. On the **Networking** tab, ensure **Enable public access** is turned **On**. This is required for the GPT to connect.
4. Complete the deployment and click **Go to Resource** to open your new Function App.

> **Note**: The deployment may error on the first attempt. If so, simply retry the creation process.

## Step 2: Configure Authentication (OAuth)

Your function needs OAuth protection to securely connect to GPT Actions.

1. In your Function App's left-hand menu, go to **Settings** > **Authentication**.
2. Click **Add identity provider**.
3. Select **Microsoft** as the provider and **Workforce** as the tenant type.
4. Choose to **Create a new app registration**. This simplifies setup by automatically configuring callback URLs.
5. On the **Permissions** tab, add the Microsoft Graph permissions `Files.Read.All` and `Sites.Read.All`. (You can skip this if you are not building a SharePoint integration).
6. Click **Add**.

## Step 3: Configure the Enterprise Application

You must grant the new application permission to impersonate the user.

1. Navigate to the **Enterprise Application** you just created (leave the Function App page).
2. Go to **API permissions** > **Add a permission**.
3. Search for **Microsoft Azure App Service** under **APIs my organization uses**.
4. Select the `user_impersonation` permission and add it.
5. An Azure admin must **Grant admin consent** for this permission.
6. Go to **Expose an API** and copy the generated scope (e.g., `api://<uuid>/user_impersonation`). Save this as `SCOPE`.
7. Go to **Authentication** > **Web** and add the Postman test redirect URI: `https://oauth.pstmn.io/v1/callback`.
8. From the application's **Overview** page, copy the **Application (client) ID** and **Directory (tenant) ID**. Save these as `CLIENT_ID` and `TENANT_ID`.

## Step 4: Create a Test HTTP Trigger Function

You'll create a simple function to verify your setup.

1. Return to your **Function App** in the Azure Portal.
2. Click **Create Function**.
3. Select **HTTP trigger** as the template.
4. Choose your preferred **Authorization level** (Function, Admin, etc.).
5. Click **Create**. Refresh the page if the function doesn't appear immediately.
6. Click into your new function and select **Get Function URL**. Copy and save this URL for testing.
7. Back in the Function App, go to **Configuration**.
8. Find the `MICROSOFT_PROVIDER_AUTHENTICATION_SECRET` variable, show its value, and copy it. Save this as your `CLIENT_SECRET`.

You should now have five key values: `CLIENT_ID`, `TENANT_ID`, `CLIENT_SECRET`, `SCOPE`, and your `FUNCTION_URL`.

## Step 5: Test OAuth with Postman

Validate your authentication flow before connecting to ChatGPT.

1. Open Postman and create a new request to your `FUNCTION_URL`.
2. Configure the **Authorization** tab:
    * **Type**: OAuth 2.0
    * **Grant Type**: Authorization Code
    * **Callback URL**: `https://oauth.pstmn.io/v1/callback`
    * **Auth URL**: `https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/authorize`
    * **Access Token URL**: `https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token`
    * **Client ID**: Your `CLIENT_ID`
    * **Client Secret**: Your `CLIENT_SECRET`
    * **Scope**: Your `SCOPE`
    * **Client Authentication**: Send client credentials in body
3. Click **Get New Access Token** and complete the Microsoft login.
4. Send the request. A successful response will be: `"This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response."`

## Step 6: Develop Your Application Logic

Replace the test function with your specific application logic. This step is unique to your use case. For an example, refer to the [SharePoint Cookbook](https://cookbook.openai.com/examples/chatgpt/gpt_actions_library/gpt_action_sharepoint_doc).

## Step 7: Connect to ChatGPT

Finally, you will connect your secured Azure Function as a GPT Action.

1. **Generate an OpenAPI schema** for your function's endpoint. Below is a template. Replace the placeholders (`{}`) with your details, including your function's specific path and operation details.

```yaml
openapi: 3.1.0
info:
  title: {insert title}
  description: {insert description}
  version: 1.0.0
servers:
  - url: https://{your_function_app_name}.azurewebsites.net/api
    description: {insert description}
paths:
  /{your_function_name}?code={enter_your_specific_endpoint_id_here}:
    post:
      operationId: {insert operationID}
      summary: {insert summary}
      requestBody:
        # ... rest of your application-specific schema
```

2. In the ChatGPT interface, create a new **Custom GPT** and navigate to **Configure** > **Actions**.
3. Paste your OpenAPI schema into the schema editor.
4. Click on **Authentication** and select **OAuth**.
5. Enter the same OAuth credentials you used in Postman:
    * **Client ID**: `CLIENT_ID`
    * **Client Secret**: `CLIENT_SECRET`
    * **Authorization URL**: `https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/authorize`
    * **Token URL**: `https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token`
    * **Scope**: `SCOPE`
6. Save the action. ChatGPT will generate a **Callback URL**. Copy this URL.
7. Return to your Enterprise Application in the Azure Portal (**Authentication** > **Web**).
8. Add a new **Redirect URI** and paste the Callback URL from ChatGPT. Click **Save**.
9. Your GPT Action is now configured. Test it within the ChatGPT interface.

## Conclusion

You have successfully set up an Azure Function as OAuth-protected middleware for a GPT Action. This pattern allows you to build powerful, secure integrations between ChatGPT and your backend services. For application-specific instructions (like custom GPT instructions), refer to the documentation for your particular use case.