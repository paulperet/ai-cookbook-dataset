# Gemini API OAuth Authentication Guide

## Overview
This guide demonstrates how to authenticate with the Gemini API using OAuth 2.0. While API keys are simpler for beginners, OAuth is required for advanced features like model tuning and semantic retrieval. This tutorial provides a simplified approach suitable for testing environments.

**Important**: For production applications, review Google's [authentication and authorization documentation](https://developers.google.com/workspace/guides/auth-overview) and [choose appropriate access credentials](https://developers.google.com/workspace/guides/create-credentials#choose_the_access_credential_that_is_right_for_you).

## Prerequisites

Before starting, ensure you have:
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk) installed locally
- A Google Cloud project (created automatically if you made an API key in [Google AI Studio](https://aistudio.google.com/app/apikey))

## Part 1: Configure Your Google Cloud Project

### Step 1: Enable the Gemini API
1. Navigate to the [Google Cloud Console](https://console.cloud.google.com/flows/enableapi?apiid=generativelanguage.googleapis.com)
2. Enable the **Google Generative Language API**
   - *Note*: If you created an API key in AI Studio, this step was completed automatically

### Step 2: Configure OAuth Consent Screen
1. Go to **Menu** > **APIs & Services** > **OAuth consent screen**
2. Select **External** as the user type and click **Create**
3. Complete the app registration form (minimal information required) and click **Save and Continue**
4. Skip adding scopes for now (click **Save and Continue**)
5. Add test users:
   - Click **Add users** under **Test users**
   - Enter your email address and any other authorized test users
   - Click **Save and Continue**
6. Review your app registration and click **Back to Dashboard**

### Step 3: Create OAuth Client ID for Desktop Application
1. Navigate to **Menu** > **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth client ID**
3. Select **Desktop app** as the application type
4. Enter a name for your credential (visible only in Google Cloud Console)
5. Click **Create**
6. Download the JSON file (named `client_secret_<identifier>.json`)

## Part 2: Set Up Application Default Credentials

This guide uses [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) for authentication.

### Option A: Using Google Colab Secrets (Recommended for Colab)

If you frequently use OAuth with Gemini API in Google Colab, store your client secret in Colab's Secrets manager:

1. Open your Google Colab notebook and click the 🔑 **Secrets** tab
2. Create a new secret named `CLIENT_SECRET`
3. Copy the contents of your `client_secret.json` file into the `Value` field
4. Toggle the notebook access permission

Now create the credential file programmatically:

```python
from google.colab import userdata
import pathlib

pathlib.Path('client_secret.json').write_text(userdata.get('CLIENT_SECRET'))
```

### Step 4: Generate Application Default Credentials

Run this command to convert your client secret into usable credentials. You'll need to do this once per Colab notebook or runtime:

```bash
gcloud auth application-default login \
  --no-browser --client-id-file client_secret.json \
  --scopes https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning,https://www.googleapis.com/auth/generative-language.retriever
```

**Important Notes:**
- Carefully follow the printed instructions (don't just click the link)
- Ensure your local `gcloud` version matches the version in Google Colab
- You'll see a "Google hasn't verified this app" dialog - this is normal, choose **Continue**
- The scopes shown are comprehensive; for production, use only the scopes your app needs (check the [API reference](https://ai.google.dev/api/rest/v1beta/tunedModels/create#authorization-scopes) for required scopes)

## Part 3: Using the Python SDK with OAuth

### Step 5: Install the Python SDK

```bash
pip install -U "google-generativeai>=0.7.2"
```

### Step 6: Test Authentication

The Python SDK automatically detects and uses application default credentials:

```python
import google.generativeai as genai

# No need to call genai.configure() with API key
print('Available base models:', [m.name for m in genai.list_models()])
```

## Appendix A: Making Authenticated REST Calls

While the Python SDK is recommended, you can make direct REST calls for debugging:

```python
import requests

# Get access token
access_token = !gcloud auth application-default print-access-token

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {access_token[0]}',
}

# List available models
response = requests.get('https://generativelanguage.googleapis.com/v1/models', headers=headers)
response_json = response.json()

for model in response_json['models']:
    print(model['name'])
```

### Sharing a Tuned Model (Beta Feature)

Some beta APIs may not be in the Python SDK yet. Here's how to share a tuned model via REST:

```python
import requests

# Set your model name and recipient email
model_name = 'your-tuned-model-name'
email_address = 'recipient@example.com'

# Get access token
access_token = !gcloud auth application-default print-access-token

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {access_token[0]}',
}

body = {
    'granteeType': 'USER',  # Options: 'USER', 'GROUP', or 'EVERYONE'
    'emailAddress': email_address,  # Optional if granteeType is 'EVERYONE'
    'role': 'READER'
}

response = requests.post(
    f'https://generativelanguage.googleapis.com/v1beta/tunedModels/{model_name}/permissions',
    json=body,
    headers=headers
)

print(response.json())
```

## Appendix B: Using Service Account Authentication

[Service accounts](https://cloud.google.com/iam/docs/service-account-overview) authenticate applications without human involvement. **Important**: Service account keys pose security risks - review [best practices](https://cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) before using in production.

### Step 1: Create a Service Account
Follow Google's guide to [create a service account](https://cloud.google.com/iam/docs/service-accounts-create#creating).

### Step 2: Create a Service Account Key
Create a key following the [service account key creation guide](https://cloud.google.com/iam/docs/keys-create-delete#creating).

### Step 3: Add Key to Colab Secrets
1. Open Colab's 🔑 **Secrets** tab
2. Create a secret named `SERVICE_ACCOUNT_KEY`
3. Paste your service account key JSON content
4. Enable notebook access

### Step 4: Authenticate with Service Account

```python
import google.generativeai as genai
import pathlib
from google.colab import userdata
from google.oauth2 import service_account

# Load service account key
pathlib.Path('service_account_key.json').write_text(userdata.get('SERVICE_ACCOUNT_KEY'))

# Create credentials with appropriate scopes
credentials = service_account.Credentials.from_service_account_file('service_account_key.json')
scoped_credentials = credentials.with_scopes([
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/generative-language.retriever'
])

# Configure Gemini API with service account credentials
genai.configure(credentials=scoped_credentials)

print('Available base models:', [m.name for m in genai.list_models()])
```

## Summary

You've now learned how to:
1. Configure OAuth for the Gemini API in Google Cloud Console
2. Set up application default credentials for testing
3. Use OAuth authentication with the Python SDK
4. Make authenticated REST calls
5. Use service accounts for application-level authentication

Remember to use the minimal required scopes for production applications and follow security best practices when handling credentials.