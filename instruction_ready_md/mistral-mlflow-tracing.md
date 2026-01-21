# Observability with Mistral AI and MLflow: A Step-by-Step Guide

This guide demonstrates how to leverage MLflow's auto-tracing capabilities to monitor and debug calls made to the Mistral AI API. By the end, you will have a functional setup that automatically logs all interactions with Mistral's models into MLflow for easy inspection.

## Prerequisites

Before you begin, ensure you have:
1. A valid `MISTRAL_API_KEY` from Mistral AI.
2. Python installed on your system.

## Step 1: Environment Setup

First, install the required Python packages. Run the following commands in your terminal or notebook cell.

```bash
pip install mistralai==1.5.0
pip install mlflow==2.20.1
```

## Step 2: Import Libraries and Configure Auto-Logging

Create a new Python script or notebook. Start by importing the necessary libraries and activating MLflow's auto-logging for Mistral AI. This single call instructs MLflow to automatically capture all subsequent Mistral API calls.

```python
import os
from mistralai import Mistral
import mlflow

# Enable automatic tracing for all Mistral AI operations
mlflow.mistral.autolog()
```

## Step 3: Initialize the Mistral Client

Initialize the Mistral client using your API key. It's best practice to store sensitive keys as environment variables.

```python
# Ensure your MISTRAL_API_KEY is set in your environment
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

## Step 4: Make an API Call

Now, use the client to send a prompt to a Mistral model. The `mlflow.mistral.autolog()` call ensures this interaction is automatically traced.

```python
# Send a chat completion request
chat_response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": "Who is the best French painter? Answer in one short sentence.",
        },
    ],
)

# Print the model's response
print(chat_response.choices[0].message.content)
```

After executing this code, you should see the model's answer printed. More importantly, a trace of this call has been automatically logged by MLflow.

## Step 5: Launch the MLflow UI to View Traces

To inspect the logged traces, you need to start the MLflow UI.

1.  Open a new terminal window.
2.  Navigate to the directory where you ran your Python code.
3.  Run the following command:

```bash
mlflow ui
```

This starts a local web server, typically at `http://localhost:5000`.

## Step 6: Inspect the Trace

1.  Open your web browser and go to `http://localhost:5000`.
2.  In the MLflow UI, navigate to the **"Traces"** section.
3.  You will see a list of traced executions. Click on the trace corresponding to your recent `chat.complete` call.

Here you can explore detailed information about the trace, including:
*   The input prompt sent to the model.
*   The model's output response.
*   Latency and timing information.
*   Any internal steps or token usage details provided by the API.

## Conclusion

You have successfully set up automatic tracing for Mistral AI using MLflow. This workflow provides immediate observability into your LLM calls, which is crucial for debugging, performance monitoring, and auditing interactions in production applications. You can now extend this by making more complex API calls or integrating tracing into larger applications.