# Gemini API Authentication Guide

This guide walks you through authenticating with the Gemini API using API keys. You'll learn how to create and securely store your API key, then use it with both the Python SDK and command-line tools.

## Prerequisites

Before you begin, you'll need:
- A Google AI Studio account
- Your Gemini API key

## Step 1: Create Your API Key

1. Navigate to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click to create a new API key with a single click
3. **Important:** Treat your API key like a password. Never commit it to version control or share it publicly.

## Step 2: Secure Storage Options

Choose the storage method that matches your development environment:

### Option A: Google Colab Secrets (Recommended for Colab Users)

If you're using Google Colab, store your key securely in Colab Secrets:

1. Open your Google Colab notebook
2. Click the 🔑 **Secrets** tab in the left panel
3. Create a new secret named `GOOGLE_API_KEY`
4. Paste your API key into the `Value` field
5. Toggle the button to allow all notebooks access to this secret

### Option B: Environment Variables (Recommended for Local Development)

For local development environments or terminal usage, store your key in an environment variable:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## Step 3: Install the Python SDK

Install the Gemini Python SDK using pip:

```bash
pip install -qU 'google-genai>=1.0.0'
```

## Step 4: Configure the SDK with Your API Key

### For Colab Users (Using Secrets)

```python
from google import genai
from google.colab import userdata

# Retrieve the API key from Colab Secrets
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### For Local Development (Using Environment Variables)

```python
import os
from google import genai

# Read the API key from environment variable
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
```

### Automatic Detection

The client can also automatically detect your API key if it's available in the environment:

```python
from google import genai

# Client will automatically look for GOOGLE_API_KEY in environment
client = genai.Client()
```

## Step 5: Choose a Model

The Gemini API offers various models optimized for different use cases. Select one that fits your needs:

```python
# Available model options (choose one):
# - gemini-2.5-flash-lite
# - gemini-2.5-flash
# - gemini-2.5-pro
# - gemini-2.5-flash-preview
# - gemini-3-pro-preview

MODEL_ID = "gemini-2.5-flash"  # Example selection
```

## Step 6: Make Your First API Call

Now you're ready to call the Gemini API. Here's a complete example:

```python
from google import genai
import os

# Initialize client (using environment variable)
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

# Choose your model
MODEL_ID = "gemini-2.5-flash"

# Make an API call
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Please give me python code to sort a list."
)

# Display the response
print(response.text)
```

## Step 7: Using cURL (Command Line)

If you prefer using the command line or need to integrate with shell scripts, you can use cURL:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{
          "text": "Please give me Python code to sort a list."
        }]
      }]
    }'
```

## Best Practices and Security Tips

1. **Never hardcode API keys** in your source files
2. **Use environment variables** or secure secret managers for production
3. **Rotate your API keys** periodically for enhanced security
4. **Monitor your API usage** through Google AI Studio dashboard
5. **Set up budget alerts** to prevent unexpected charges

## Next Steps

Now that you're authenticated, you can:
- Explore the [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- Check out the [quickstart guides](https://github.com/google-gemini/cookbook/tree/main/quickstarts)
- Learn about different [Gemini models](https://ai.google.dev/gemini-api/docs/models) and their capabilities

## Troubleshooting

If you encounter authentication issues:

1. **Verify your API key** is correct and hasn't expired
2. **Check environment variables** with `echo $GOOGLE_API_KEY`
3. **Ensure proper installation** of the Python SDK
4. **Review API quotas** in Google AI Studio
5. **Check network connectivity** and firewall settings

Remember: Your API key is the gateway to the Gemini API. Keep it secure, and you'll be ready to build amazing AI-powered applications!