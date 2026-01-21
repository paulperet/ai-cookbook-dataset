# Gemini API: LLM Observability with MLflow

## Overview

[MLflow](https://mlflow.org/) is an open-source platform designed to manage the complexities of the machine learning lifecycle. Its [MLflow Tracing](https://mlflow.org/docs/latest/tracing/) feature enhances observability in Generative AI applications by capturing detailed execution data for each step of a request. This includes inputs, outputs, and metadata, making it easier to debug and understand application behavior.

This guide demonstrates the basic usage of MLflow's built-in integration with the Google Gen AI SDK (`google-genai`) to instrument your Gemini API calls.

## Prerequisites

Before you begin, ensure you have:
1.  A Google AI Studio API key. You can create one at [AI Studio](https://aistudio.google.com/app/apikey).
2.  A Databricks workspace for the MLflow tracking server (a free trial is used in this guide).

## Setup

First, install the required packages. The integration with `google-genai` requires MLflow version 2.20.3 or higher.

```bash
pip install -q google-genai "mlflow>=2.20.3"
```

Now, import the necessary libraries and configure your Gemini client with your API key.

```python
import google.genai as genai
import os
import mlflow

# Initialize the Gemini client
# Replace 'YOUR_API_KEY' with your actual key or use a secure method to load it.
client = genai.Client(api_key='YOUR_API_KEY')
```

## Step 1: Configure the MLflow Tracking Server

This example uses the free Databricks trial as a managed MLflow tracking server, which is easy to connect from environments like Colab.

1.  Create a Databricks account via the [Databricks Trial Signup Page](http://signup.databricks.com/).
2.  Generate a Personal Access Token (PAT) for your workspace by following [this guide](https://docs.databricks.com/aws/en/dev-tools/auth/pat).
3.  Set the following environment variables with your workspace details:
    *   `DATABRICKS_HOST`: Your workspace URL (e.g., `https://<your-workspace-host>.cloud.databricks.com/`)
    *   `DATABRICKS_TOKEN`: Your personal access token.

```python
# Set your Databricks workspace credentials (load securely in production)
WORKSPACE_EMAIL = "your.email@example.com"  # Your account email
WORKSPACE_PAT = "your_databricks_pat"       # Your personal access token
WORKSPACE_URL = "your_databricks_host_url"  # Your workspace host URL

# Configure MLflow to use the Databricks tracking server
mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks-uc')
os.environ['DATABRICKS_HOST'] = WORKSPACE_URL
os.environ['DATABRICKS_TOKEN'] = WORKSPACE_PAT

# Log in to the MLflow server
mlflow.login()

# Create and set an experiment for your Gemini traces
mlflow.create_experiment(
    f"/Users/{WORKSPACE_EMAIL}/gemini",
    artifact_location="dbfs:/Volumes/workspace/default/gemini",
)
mlflow.set_experiment(f"/Users/{WORKSPACE_EMAIL}/gemini")
```

## Step 2: Enable Auto-Logging for Gemini

MLflow can automatically capture traces from your Gemini API calls. Enable this feature with a single function call. Auto-logging captures prompts, completions, latencies, model parameters, function calls, and any exceptions.

```python
# Enable automatic tracing for all Gemini SDK calls
mlflow.gemini.autolog()
```

## Step 3: Trace a Simple Content Generation Call

With auto-logging enabled, any call to the Gemini SDK will be traced. Let's start with a simple text generation request.

```python
MODEL_ID = "gemini-3-flash-preview"  # You can change this to any supported model

result = client.models.generate_content(
    model=MODEL_ID,
    contents="The opposite of hot is"
)
print(result.text)
```

**Expected Output:**
```
cold
```

The trace for this call, including the prompt, response, and model metadata, is now logged to your MLflow experiment. You can view it in the MLflow UI.

## Step 4: Trace a Multi-Turn Chat Interaction

MLflow tracing captures the structure of conversational interactions. The following example shows a multi-turn chat within a named span for better organization in the trace view.

```python
with mlflow.start_span(name="multi-turn"):
    chat = client.chats.create(model=MODEL_ID)
    response = chat.send_message("In one sentence, explain how a computer works to a young child.")
    print(response.text)
    response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")
    print(response.text)
```

**Expected Output:**
```
A computer follows instructions, like a recipe, to do things with numbers and pictures.
A computer works by executing a sequence of instructions, called a program, written in a language it understands. These instructions manipulate binary data (0s and 1s) representing information, performing calculations, storing and retrieving data from memory and storage devices, and interacting with input and output devices like the keyboard, screen, and network to complete tasks.
```

The trace will show the entire chat session, with each turn captured as a child span under the "multi-turn" parent span.

## Step 5: Trace Function Calling

If your application uses [Gemini's function calling](https://ai.google.dev/gemini-api/docs/function-calling), MLflow automatically captures the function definitions, the model's function call request, and the function's response.

First, define the tools (functions) you want to make available to the model.

```python
def add(a: float, b: float):
    """returns a + b."""
    return a + b

def subtract(a: float, b: float):
    """returns a - b."""
    return a - b

def multiply(a: float, b: float):
    """returns a * b."""
    return a * b

def divide(a: float, b: float):
    """returns a / b."""
    return a / b

operation_tools = [add, subtract, multiply, divide]
```

Next, create a chat session with these tools enabled and send a message that should trigger a function call.

```python
chat = client.chats.create(
    model=MODEL_ID,
    config={
        "tools": operation_tools,
    }
)

response = chat.send_message(
    "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
)
print(response.text)
```

**Expected Output:**
```
2508
```

The MLflow trace for this interaction will include the tool definitions, the model's decision to call the `multiply` function with arguments `a=57` and `b=44`, and the function's returned result.

## Conclusion

You have successfully set up MLflow Tracing to monitor your Gemini API calls. This provides crucial observability for debugging and understanding the behavior of your Generative AI applications.

To explore more advanced configurations and features, visit the official documentation:
*   [MLflow Gemini Tracing Integration](https://mlflow.org/docs/latest/tracing/integrations/gemini)
*   [MLflow Tracing Overview](https://mlflow.org/docs/latest/tracing/)