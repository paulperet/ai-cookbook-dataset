# Azure AI Inference with Phi-4 Models: A Step-by-Step Guide

This guide demonstrates how to use the Azure AI Inference SDK to access Azure-hosted models, specifically Microsoft's Phi-4 models. You will learn how to set up your environment, authenticate, and make completion calls using the latest SDK methods.

> **Important Note:** This guide uses the latest Azure AI Inference SDK. The correct pattern for making completion calls is through the `ChatCompletionsClient` with appropriate message objects.

## Prerequisites & Setup

Before you begin, ensure you have the necessary packages installed and your environment configured.

### 1. Install Required Packages

Open your terminal or command prompt and run the following command to install the required Python libraries:

```bash
pip install azure-ai-inference python-dotenv
```

### 2. Configure Your Environment Variables

You need to store your Azure AI credentials securely. Create a file named `local.env` in your project directory and add the following variables:

```bash
# Azure AI Configuration
AZURE_API_KEY=your_azure_api_key_here
AZURE_ENDPOINT=your_azure_endpoint_here
AZURE_MODEL=Phi-4-reasoning
```

**Instructions:**
1.  Replace `your_azure_api_key_here` with your actual Azure API key.
2.  Replace `your_azure_endpoint_here` with your Azure AI endpoint URL.
3.  You can change the `AZURE_MODEL` value to another deployed model name, such as `Phi-4-mini-reasoning`.

> **Security Note:** Keep your `local.env` file secure. Add it to your `.gitignore` file to prevent accidentally committing your API key to a public repository.

## Step 1: Load Environment Variables and Initialize the Client

Now, let's write the Python code to load your credentials and create the Azure AI client.

1.  Import the necessary modules.
2.  Load the environment variables from your `local.env` file.
3.  Initialize the `ChatCompletionsClient` with your endpoint and API key.

```python
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load variables from local.env file
load_dotenv('local.env')

# Access the environment variables
api_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
model_name = os.getenv("AZURE_MODEL", "Phi-4")  # Default to "Phi-4" if not set

# Validate that required variables are available
if not api_key or not endpoint:
    raise ValueError("AZURE_API_KEY and AZURE_ENDPOINT must be set in the local.env file.")

print(f"Azure AI Endpoint: {endpoint}")
print(f"Azure AI Model: {model_name}")

# Initialize the Azure AI Inference client
try:
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    print("Azure AI Inference client initialized successfully")
except Exception as e:
    print(f"Error initializing client: {str(e)}")
    print("Please check your endpoint URL and API key.")
```

## Step 2: Create Helper Functions for Model Inference

To simplify interactions with the model, we'll create two helper functions. These functions handle the API calls and provide helpful error messages.

### Function 1: `generate_chat_completion`
This function sends a structured conversation (a list of messages) to the model and returns its response. It includes logic to try alternative model names if the primary one fails.

### Function 2: `generate_completion`
This is a convenience wrapper for sending a single user prompt, which internally calls `generate_chat_completion`.

```python
def generate_chat_completion(messages, temperature=0.7, max_tokens=20000, model=model_name):
    """Generate a completion using Azure AI Inference SDK's chat completions API"""
    # List of model names to try in order if the primary model fails
    model_alternatives = [model, "Phi-4", "phi-4"]

    for attempt_model in model_alternatives:
        try:
            print(f"Making API call with model: {attempt_model}")

            # Convert message dictionaries to proper SDK message objects
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    formatted_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    formatted_messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AssistantMessage(content=msg["content"]))

            # Call the Azure AI Inference API
            response = client.complete(
                model=attempt_model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"Success with model: {attempt_model}")
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            print(f"Error during API call with {attempt_model}: {error_message}")

            # If this is the last model in our list, provide detailed error information
            if attempt_model == model_alternatives[-1]:
                print("\nAll model attempts failed. Please check:")
                print("1. Your model deployment names in Azure AI Studio")
                print("2. Your API key and endpoint URL")
                print("3. That the models are actually deployed to your Azure AI resource")
                print("\nYou can modify your local.env file to use the correct model name.")
                return None

            # If it's a "model not found" error, try the next alternative
            if "NOT FOUND" in error_message or "model not found" in error_message.lower():
                print(f"Model {attempt_model} not found, trying alternative...")
                continue
            else:
                # For other errors, return the error message
                return f"Error: {error_message}"

    return None

def generate_completion(prompt, temperature=0.7, max_tokens=10000, model=model_name):
    """Generate a text completion using a simple user prompt"""
    # Use the chat completion function with a single user message
    return generate_chat_completion([{"role": "user", "content": prompt}], temperature, max_tokens, model)
```

## Step 3: Run Your First Inference

Let's test the setup with a simple, somewhat whimsical query. This will confirm that your client is correctly configured and can communicate with the Azure AI service.

We will construct a conversation with a system message (to set the AI's behavior) and a user message (our question).

```python
# Define the conversation messages
example_messages = [
    {"role": "system", "content": "You are a helpful AI assistant that answers questions accurately and concisely."},
    {"role": "user", "content": "How many strawberries do I need to collect 9 r's?"}
]

print("Sending the following messages to the model:")
for msg in example_messages:
    print(f"{msg['role'].title()}: {msg['content']}")
print("\n--- Generating Response ---\n")

# Call the helper function to get a response from the model
response = generate_chat_completion(example_messages)

print("Model Response:")
print(response)
```

When you run this code, the model will process the query. It recognizes the puzzle-like nature of the question—that the word "strawberry" contains three 'r's. Therefore, to collect nine 'r's, you would need three strawberries.

**Expected Output:**
```
Sending the following messages to the model:
System: You are a helpful AI assistant that answers questions accurately and concisely.
User: How many strawberries do I need to collect 9 r's?

--- Generating Response ---

Making API call with model: Phi-4-reasoning
Success with model: Phi-4-reasoning
Model Response:
The word "strawberry" contains 3 r's. To collect 9 r's, you would need 3 strawberries (since 3 x 3 = 9).
```

## Summary

You have successfully:
1.  Configured your environment with Azure AI credentials.
2.  Initialized the Azure AI Inference SDK client.
3.  Created reusable helper functions for chat and text completions.
4.  Made your first API call to a Phi-4 model on Azure.

You can now use the `generate_chat_completion` function for more complex conversational tasks or `generate_completion` for straightforward prompts. Experiment by adjusting parameters like `temperature` (for creativity) and `max_tokens` (for response length).