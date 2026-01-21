# Evaluating AI Agents with Langfuse and the OpenAI Agents SDK

This guide demonstrates how to monitor, trace, and evaluate AI agents built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) using [Langfuse](https://langfuse.com/docs). You will learn to instrument your agent for observability and implement both online and offline evaluation strategies to ensure reliability and performance in production.

## Prerequisites

Before you begin, ensure you have the following:
- An [OpenAI API key](https://platform.openai.com/api-keys).
- A [Langfuse account](https://langfuse.com/) and project credentials (Public Key, Secret Key).

## Setup

Install the required Python libraries.

```bash
pip install openai-agents nest_asyncio "pydantic-ai[logfire]" langfuse datasets
```

## Step 1: Configure Langfuse and OpenTelemetry Instrumentation

First, set your environment variables to connect to Langfuse and configure OpenTelemetry to send traces.

```python
import os
import base64

# Get keys from your Langfuse project settings: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..." 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

# Build Basic Auth header for OpenTelemetry.
LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()
 
# Configure OpenTelemetry endpoint & headers
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST") + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```

Initialize the Langfuse client to verify the connection.

```python
from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
```

## Step 2: Instrument the OpenAI Agents SDK

We'll use Pydantic Logfire's instrumentation to automatically send traces from the OpenAI Agents SDK to Langfuse via OpenTelemetry.

```python
import nest_asyncio
import logfire

# Required for running async code in notebooks/scripts
nest_asyncio.apply()

# Configure logfire instrumentation.
logfire.configure(
    service_name='my_agent_service',
    send_to_logfire=False, # We send to Langfuse, not Logfire
)
# This method automatically patches the OpenAI Agents SDK.
logfire.instrument_openai_agents()
```

## Step 3: Test Basic Instrumentation

Let's run a simple agent to confirm traces are being sent to Langfuse.

```python
import asyncio
from agents import Agent, Runner

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a senior software engineer",
    )

    result = await Runner.run(agent, "Tell me why it is important to evaluate AI agents.")
    print(result.final_output)

# Run the async function
asyncio.run(main())

# Ensure all pending events are sent
langfuse.flush()
```

**Expected Output:**
```
Evaluating AI agents is crucial for debugging failures, monitoring costs and performance in real-time, and improving reliability and safety through continuous feedback.
```

Check your [Langfuse Traces Dashboard](https://cloud.langfuse.com/traces) to confirm the trace was recorded. You should see a span for the agent run and sub-spans for the underlying LLM calls.

## Step 4: Observe a More Complex Agent with Tools

Now, let's create an agent with a function tool to see a more detailed trace structure.

```python
import asyncio
from agents import Agent, Runner, function_tool

# Define a simple mock tool
@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)

async def main():
    result = await Runner.run(agent, input="What's the weather in Berlin?")
    print(result.final_output)

asyncio.run(main())
```

**Expected Output:**
```
The weather in Berlin is sunny.
```

In Langfuse, the trace will now contain distinct spans for:
1. The overall agent run.
2. The tool call to `get_weather`.
3. The LLM calls (Responses API).

This breakdown allows you to inspect token usage, latency, and costs for each component.

## Step 5: Implement Online Evaluation Metrics

Online evaluation involves monitoring your agent in a live environment. Key metrics include costs, latency, user feedback, and automated LLM-as-a-Judge scoring.

### 5.1 Enrich Traces with Metadata

Add contextual information like `user_id`, `session_id`, and `tags` to your traces for better analysis.

```python
input_query = "Why is AI agent evaluation important?"

with langfuse.start_as_current_span(
    name="OpenAI-Agent-Trace",
    ) as span:
    
    async def main(input_query):
            agent = Agent(
                name = "Assistant",
                instructions = "You are a helpful assistant.",
            )

            result = await Runner.run(agent, input_query)
            print(result.final_output)
            return result

    result = await main(input_query)
 
    # Pass additional attributes to the span
    span.update_trace(
        input=input_query,
        output=result,
        user_id="user_123",
        session_id="my-agent-session",
        tags=["staging", "demo", "OpenAI Agent SDK"],
        metadata={"email": "user@langfuse.com"},
        version="1.0.0"
        )
 
# Flush events
langfuse.flush()
```

### 5.2 Capture User Feedback

If your agent has a user interface, you can capture explicit feedback (e.g., thumbs up/down) and attach it to the trace.

```python
from agents import Agent, Runner, WebSearchTool
import ipywidgets as widgets
from IPython.display import display
from langfuse import get_client
 
langfuse = get_client()

agent = Agent(
    name="WebSearchAgent",
    instructions="You are an agent that can search the web.",
    tools=[WebSearchTool()]
)

def on_feedback(button):
    if button.icon == "thumbs-up":
      langfuse.create_score(
            value=1,
            name="user-feedback",
            comment="The user gave this response a thumbs up",
            trace_id=trace_id
        )
    elif button.icon == "thumbs-down":
      langfuse.create_score(
            value=0,
            name="user-feedback",
            comment="The user gave this response a thumbs down",
            trace_id=trace_id
        )
    print("Scored the trace in Langfuse")

user_input = input("Enter your question: ")

# Run agent
with langfuse.start_as_current_span(
    name="OpenAI-Agent-Trace",
    ) as span:
    
    result = Runner.run_sync(agent, user_input)
    print(result.final_output)
    trace_id = langfuse.get_current_trace_id()

    span.update_trace(
        input=user_input,
        output=result.final_output,
    )

# Get feedback
print("How did you like the agent response?")

thumbs_up = widgets.Button(description="👍", icon="thumbs-up")
thumbs_down = widgets.Button(description="👎", icon="thumbs-down")

thumbs_up.on_click(on_feedback)
thumbs_down.on_click(on_feedback)

display(widgets.HBox([thumbs_up, thumbs_down]))

langfuse.flush()
```

### 5.3 Implement LLM-as-a-Judge

Automatically evaluate your agent's output for criteria like correctness or toxicity using another LLM as a judge. This can be set up as a separate span in your trace.

```python
from agents import Agent, Runner, WebSearchTool

agent = Agent(
    name="WebSearchAgent",
    instructions="You are an agent that can search the web.",
    tools=[WebSearchTool()]
)

input_query = "Is eating carrots good for the eyes?"

# Run agent
with langfuse.start_as_current_span(name="OpenAI-Agent-Trace") as span:
    result = Runner.run_sync(agent, input_query)

    # Add input and output values to parent trace
    span.update_trace(
        input=input_query,
        output=result.final_output,
    )
    # Here you would typically add another span/LLM call to act as the "judge"
```

## Step 6: Perform Offline Evaluation with a Dataset

Offline evaluation uses a static dataset to benchmark your agent's performance before deployment.

### 6.1 Load a Benchmark Dataset

We'll use the `search-dataset` from Hugging Face, which contains questions and expected answers.

```python
import pandas as pd
from datasets import load_dataset

# Fetch search-dataset from Hugging Face
dataset = load_dataset("junzhang1207/search-dataset", split = "train")
df = pd.DataFrame(dataset)
print("First few rows of search-dataset:")
print(df.head())
```

### 6.2 Create a Dataset in Langfuse

Upload your dataset items to Langfuse to manage evaluation runs.

```python
from langfuse import get_client
langfuse = get_client()

langfuse_dataset_name = "search-dataset_huggingface_openai-agent"

# Create a dataset in Langfuse
langfuse.create_dataset(
    name=langfuse_dataset_name,
    description="search-dataset uploaded from Huggingface",
    metadata={
        "date": "2025-03-14",
        "type": "benchmark"
    }
)

# Upload the first 50 items
for idx, row in df.iterrows():
    langfuse.create_dataset_item(
        dataset_name=langfuse_dataset_name,
        input={"text": row["question"]},
        expected_output={"text": row["expected_answer"]}
    )
    if idx >= 49:
        break
```

### 6.3 Run the Agent on the Dataset

Define a function to run your agent on each dataset item and link the results back to Langfuse.

```python
from agents import Agent, Runner, WebSearchTool
from langfuse import get_client
 
langfuse = get_client()
dataset_name = "search-dataset_huggingface_openai-agent"
current_run_name = "qna_model_v3_run_05_20" # Identifies this specific evaluation run

agent = Agent(
    name="WebSearchAgent",
    instructions="You are an agent that can search the web.",
    tools=[WebSearchTool(search_context_size= "high")]
)
 
def run_openai_agent(question):
    with langfuse.start_as_current_generation(name="qna-llm-call") as generation:
        result = Runner.run_sync(agent, question)
 
        generation.update_trace(
            input= question,
            output=result.final_output,
        )
        return result.final_output
 
dataset = langfuse.get_dataset(name=dataset_name)
 
for item in dataset.items:
    # Execute the agent for each dataset item
    output = run_openai_agent(item.input["text"])
    # Link the trace to the dataset item for evaluation
    langfuse.create_dataset_run_item(
        dataset_item_id=item.id,
        run_name=current_run_name,
        input=item.input,
        output={"text": output}
    )
```

You can now review the results in the Langfuse UI, compare outputs against expected answers, and calculate performance metrics.

## Summary

You have successfully set up Langfuse to trace and evaluate an AI agent built with the OpenAI Agents SDK. You can now:

*   **Monitor** internal steps, costs, and latency in real-time.
*   **Enrich** traces with user and session context.
*   **Collect** explicit user feedback and automated LLM-judge scores.
*   **Benchmark** your agent's performance using offline datasets.

This comprehensive observability and evaluation framework is essential for debugging, optimizing, and safely deploying reliable AI agents to production.