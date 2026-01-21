# Guide: Using Function Calling with Azure OpenAI

This guide demonstrates how to use the function calling capability with the Azure OpenAI service. Functions allow you to extend the model's capabilities by connecting it to external tools and data sources.

> **Note:** Chat functions require model versions beginning with `gpt-4` and `gpt-35-turbo` with the `-0613` labels. They are not supported by older model versions.

## Prerequisites

Before you begin, ensure you have:
1. An Azure OpenAI resource created in the [Azure Portal](https://portal.azure.com).
2. A deployed model (e.g., `gpt-35-turbo-0613` or `gpt-4-0613`) within your resource. You will need the deployment name.

## Setup

First, install the required Python packages and import the necessary libraries.

```bash
pip install "openai>=1.0.0,<2.0.0"
pip install python-dotenv
```

```python
import os
import openai
import dotenv

# Load environment variables from a .env file
dotenv.load_dotenv()
```

## Authentication

The Azure OpenAI service supports two primary authentication methods: API keys and Azure Active Directory (Azure AD). Choose the method that fits your security setup.

### Method 1: Authenticate with an API Key

This is the most common method. You'll need your endpoint and API key from the Azure Portal.

1.  Navigate to your Azure OpenAI resource in the [Azure Portal](https://portal.azure.com).
2.  Go to **"Keys and Endpoint"** under **"Resource Management"**.
3.  Copy your **Endpoint** and one of your **Keys**.

Set these as environment variables in a `.env` file or your system:

```bash
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key-here"
```

Now, configure the OpenAI client in your code:

```python
# Set this flag to False to use API Key authentication
use_azure_active_directory = False

if not use_azure_active_directory:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2023-09-01-preview"  # Use the latest stable API version
    )
```

### Method 2: Authenticate with Azure Active Directory

For enhanced security using managed identities or service principals, use Azure AD authentication.

1.  First, install the Azure Identity library:

    ```bash
    pip install "azure-identity>=1.15.0"
    ```

2.  Configure your environment. The client will use the `DefaultAzureCredential`, which automatically tries multiple authentication methods (like environment variables, managed identity, or Visual Studio Code login).

3.  Update your code to use the token provider:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Set this flag to True to use Azure AD authentication
use_azure_active_directory = True

if use_azure_active_directory:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2023-09-01-preview"
    )
```

> **Note:** The `AzureOpenAI` client can infer configuration from environment variables. For example, it will look for `api_key` in `AZURE_OPENAI_API_KEY` and `azure_endpoint` in `AZURE_OPENAI_ENDPOINT` if you don't provide them explicitly.

## Step 1: Define Your Function

The core of function calling is defining the capabilities you want to expose to the model. You describe each function using a JSON schema.

In this example, we'll define a simple function to get the current weather.

```python
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            "required": ["location"],  # The 'location' parameter is mandatory
        },
    }
]
```

## Step 2: Initiate the Chat with Function Definitions

Now, start a conversation with the model, providing the list of functions you defined. The model will analyze the user's request and decide if it needs to call a function.

1.  First, set your deployment name. This is the name you gave your model in the Azure OpenAI Studio.

    ```python
    deployment = "your-deployment-name-here"  # e.g., "gpt-35-turbo-0613"
    ```

2.  Create the initial conversation messages and call the Chat Completions API.

    ```python
    messages = [
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
        },
        {"role": "user", "content": "What's the weather like today in Seattle?"}
    ]

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        tools=functions,  # Pass your function definitions here
        tool_choice="auto",  # Let the model decide whether to call a function
    )
    ```

## Step 3: Handle the Function Call

If the model determines it needs to use your function, the response will contain a `tool_calls` object with the function name and its arguments.

1.  Check for a function call in the response and extract the details.

    ```python
    import json

    # Check if the model wants to call a tool/function
    if response.choices[0].message.tool_calls:
        function_call = response.choices[0].message.tool_calls[0].function
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)

        print(f"Model wants to call function: {function_name}")
        print(f"Function arguments: {function_args}")
    ```

2.  Execute the corresponding function in your code. Here is a mock implementation.

    ```python
    def get_current_weather(location, format="celsius"):
        """
        Mock function to simulate fetching weather data.
        In a real application, you would call a weather API here.
        """
        # For demonstration, return a hard-coded response
        return {
            "temperature": "22",
            "unit": format,
            "description": "Sunny",
            "location": location
        }

    # Call the function with the arguments provided by the model
    if function_name == "get_current_weather":
        function_response = get_current_weather(**function_args)
        print(f"Function returned: {function_response}")
    ```

## Step 4: Submit the Function Result and Get the Final Answer

The model needs the function's result to continue. You provide this by adding a new message with the role `"tool"` (or `"function"` in older API versions) containing the result.

1.  Append the function's response to the conversation history.

    ```python
    # Append the function's response as a new message
    messages.append(
        {
            "role": "tool",
            "tool_call_id": response.choices[0].message.tool_calls[0].id,
            "content": json.dumps(function_response),
        }
    )
    ```

    > **Note:** The `tool_call_id` must match the ID from the initial `tool_calls` object to link the response correctly.

2.  Send the updated conversation back to the model to get a final, informed answer.

    ```python
    final_response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        tools=functions,
    )

    # Print the model's final answer to the user
    print(final_response.choices[0].message.content)
    ```

The model will now use the data from `get_current_weather` to generate a coherent answer, such as *"The weather in Seattle is sunny with a temperature of 22°C."*

## Summary

You have successfully implemented the function calling pattern with Azure OpenAI:
1.  **Defined** a function with a JSON schema.
2.  **Initiated** a chat where the model could request to use that function.
3.  **Executed** the function locally when called.
4.  **Provided** the result back to the model to generate a final response.

This pattern is powerful for building agents and applications where the LLM can interact with databases, APIs, and other external systems.