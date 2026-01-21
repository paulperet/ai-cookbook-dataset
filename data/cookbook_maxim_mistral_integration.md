# Observability with Mistral AI and Maxim: A Step-by-Step Guide

This guide demonstrates how to integrate **Maxim AI** for comprehensive observability of your Mistral AI LLM calls. You'll learn to trace requests, analyze performance metrics, and visualize agent behavior with minimal code changes.

## What is Maxim?

Maxim AI provides full-stack observability for AI applications. With a simple one-line integration, you can monitor:
*   **Performance Analytics:** Track latency, token consumption, and costs.
*   **Advanced Visualization:** Understand complex agent workflows through intuitive dashboards.

## Prerequisites

Before you begin, ensure you have:
1.  A **Mistral AI API key**.
2.  A **Maxim AI account**. [Sign up here](https://getmaxim.ai/signup) to get your API key and create a Log Repository (you'll receive a Log Repository ID).

## Step 1: Install Required Packages

Install the `mistralai` client and the `maxim-py` SDK.

```bash
pip install mistralai maxim-py
```

## Step 2: Configure Your Environment

Set your API keys and Log Repository ID as environment variables. This example uses Google Colab's secrets manager, but you can adapt it for your local environment (e.g., using a `.env` file or `os.environ` directly).

```python
import os

# Retrieve your secrets (Colab-specific). For local setup, you might use:
# MAXIM_API_KEY = "your-key-here"
MAXIM_API_KEY = "your-maxim-api-key"
MAXIM_LOG_REPO_ID = "your-log-repo-id"
MISTRAL_API_KEY = "your-mistral-api-key"

# Set environment variables
os.environ["MAXIM_API_KEY"] = MAXIM_API_KEY
os.environ["MAXIM_LOG_REPO_ID"] = MAXIM_LOG_REPO_ID
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
```

## Step 3: Initialize the Maxim Logger

Create a logger instance. This object will capture and forward all telemetry data to your Maxim dashboard.

```python
from maxim import Maxim

logger = Maxim().logger()
# You will see initialization logs confirming the SDK is ready.
```

## Step 4: Make Synchronous LLM Calls with Observability

Now, you'll make your first observed LLM call. The `MaximMistralClient` wraps the standard Mistral client, automatically logging all interactions.

```python
from mistralai import Mistral
from maxim.logger.mistral import MaximMistralClient

# Use the Maxim-wrapped client within a context manager
with MaximMistralClient(Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
), logger) as mistral:

    # Make a standard chat completion call
    res = mistral.chat.complete(
        model="mistral-medium-latest",
        messages=[
            {
                "content": "Who is the best French painter? Answer in one short sentence.",
                "role": "user",
            },
        ]
    )

    # Handle the response
    print(res.choices[0].message.content)
```

**Expected Output:**
```
Claude Monet is often regarded as the best French painter, renowned for his pioneering role in Impressionism.
```

## Step 5: Make Asynchronous LLM Calls

The integration also supports asynchronous operations, which is essential for building scalable applications.

```python
import asyncio

async def make_async_call():
    async with MaximMistralClient(Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', ''),
    ), logger) as mistral:

        response = await mistral.chat.complete_async(
            model='mistral-small-latest',
            messages=[
                {
                    'role': 'user',
                    'content': 'Explain the difference between async and sync programming in Python in one sentence.'
                }
            ]
        )
        print(response.choices[0].message.content)

# Run the async function
await make_async_call()  # In a notebook
# Or use: asyncio.run(make_async_call()) in a script
```

**Expected Output:**
```
Async programming in Python allows for concurrent execution of tasks using `async` and `await` keywords, while sync programming executes tasks sequentially, blocking until each task completes.
```

## Step 6: View Your Traces in Maxim

After running the code, your observability data is automatically sent to Maxim.

1.  Log in to the [Maxim Platform](https://app.getmaxim.ai/).
2.  Navigate to the **Logs** section.
3.  Select the Log Repository you created earlier.
4.  Switch to the **`Logs`** tab to inspect the detailed traces, including latency, token usage, and the full request/response cycle.

## Next Steps

You have successfully instrumented your Mistral AI application for observability. Explore the Maxim dashboard to:
*   Set up alerts for latency or error thresholds.
*   Analyze token usage and cost trends over time.
*   Visualize the execution flow of complex, multi-step AI agents.

---
**Feedback & Support**

For feedback, bug reports, or feature requests, please create an issue on the [Maxim Docs GitHub repository](https://github.com/maximhq/maxim-docs).