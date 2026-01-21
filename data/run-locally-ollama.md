# How to Run GPT-oSS Locally with Ollama: A Step-by-Step Guide

Want to run OpenAI's **GPT-oSS** on your own hardware? This guide walks you through setting up **gpt-oSS-20b** or **gpt-oSS-120b** locally using [Ollama](https://ollama.ai). You'll learn how to chat with it offline, use its API, and connect it to the Agents SDK.

> **Note:** This guide is designed for consumer hardware (PCs, Macs). For server applications with dedicated GPUs (e.g., NVIDIA H100s), please refer to our [vLLM guide](https://cookbook.openai.com/articles/gpt-oss/run-vllm).

## Prerequisites

Before you begin, ensure you have:
1.  **Ollama** installed. [Download it here](https://ollama.com/download).
2.  A system with sufficient memory for your chosen model.

## Step 1: Choose Your Model

Ollama supports both sizes of GPT-oSS. Choose based on your hardware:

| Model | Size | Recommended Hardware | Best For |
| :--- | :--- | :--- | :--- |
| **`gpt-oSS-20b`** | Smaller | ≥16GB VRAM or unified memory | High-end consumer GPUs or Apple Silicon Macs |
| **`gpt-oSS-120b`** | Larger, full-sized | ≥60GB VRAM or unified memory | Multi-GPU or powerful workstation setups |

**Important Notes:**
*   These models are provided **MXFP4 quantized** by default.
*   You can offload computation to your CPU if VRAM is limited, but this will significantly reduce performance.

## Step 2: Download the Model

Open your terminal and run the appropriate command to pull your chosen model.

```bash
# For the 20B parameter model
ollama pull gpt-oSS:20b

# For the 120B parameter model
ollama pull gpt-oSS:120b
```

This downloads the model to your local machine.

## Step 3: Chat with the Model

You can now start an interactive chat session directly in your terminal.

```bash
ollama run gpt-oSS:20b
```

Ollama automatically applies a chat template that mimics the [OpenAI harmony format](https://cookbook.openai.com/articles/openai-harmony). Simply type your message and begin the conversation.

## Step 4: Use the Chat Completions API

Ollama exposes a Chat Completions-compatible API, allowing you to use the familiar OpenAI Python SDK with minimal changes.

First, ensure you have the `openai` package installed.

```bash
pip install openai
```

Now, you can write a script to interact with your local model.

```python
from openai import OpenAI

# Configure the client to point to your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Local Ollama API endpoint
    api_key="ollama"                       # Placeholder key (required but unused)
)

# Make a Chat Completions request
response = client.chat.completions.create(
    model="gpt-oSS:20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)

# Print the model's response
print(response.choices[0].message.content)
```

This pattern will feel instantly familiar if you've used the official OpenAI API.

> **Alternative:** You can also use the native Ollama SDKs for [Python](https://github.com/ollama/ollama-python) or [JavaScript](https://github.com/ollama/ollama-js).

## Step 5: Implement Function Calling

Ollama supports tool/function calling. Here's how you can define a tool and have the model decide when to use it.

```python
# Define the tools available to the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            },
        },
    }
]

# Make a request that might trigger the tool
response = client.chat.completions.create(
    model="gpt-oSS:20b",
    messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
    tools=tools
)

# Inspect the response to see if the model wants to call a function
print(response.choices[0].message)
```

**Key Concept:** The model may use chain-of-thought reasoning. You must handle the conversation loop: if the response contains a tool call, execute the corresponding function, then send the result back to the model in a follow-up message so it can formulate a final answer for the user.

## Step 6: Integrate with the Agents SDK

You can use your local GPT-oSS model with OpenAI's Agents SDK. The process involves configuring the SDK to use Ollama as its backend via a proxy.

### Python Agents SDK (using LiteLLM)

This example shows how to create an agent that uses the local model via LiteLLM.

```python
import asyncio
from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

# Disable tracing for simplicity in this example
set_tracing_disabled(True)

# 1. Define a tool for the agent to use
@function_tool
def get_weather(city: str):
    """Simulates fetching weather data."""
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

# 2. Main async function to run the agent
async def main():
    # Create an agent configured to use the local Ollama model
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.", # Customize agent behavior
        model=LitellmModel(model="ollama/gpt-oSS:120b", api_key="placeholder"),
        tools=[get_weather],
    )

    # Run the agent with a user query
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

**Setup Notes:**
*   Install the required packages: `pip install agents openai-agents-litellm`.
*   The `LiteLLMModel` acts as a bridge, routing requests to your local `ollama/gpt-oSS:120b` endpoint.

### TypeScript/JavaScript Agents SDK

For the TypeScript SDK, you can use the [AI SDK](https://openai.github.io/openai-agents-js/extensions/ai-sdk/) with its [community Ollama adapter](https://ai-sdk.dev/providers/community-providers/ollama).

## (Optional) Responses API Workaround

Ollama does not natively support the newer Responses API format. If you need it, you have two options:

1.  **Use a Proxy:** Employ [Hugging Face's `responses.js` proxy](https://github.com/huggingface/responses.js) to convert Chat Completions output to the Responses API format.
2.  **Run the Example Server:** Use the basic example server provided in the GPT-oSS repository.

    ```bash
    # Install the gpt-oSS package
    pip install gpt-oSS

    # Run the server with Ollama as the backend
    python -m gpt_oSS.responses_api.serve \
        --inference_backend=ollama \
        --checkpoint gpt-oSS:20b
    ```

    > **Note:** This server is a basic example and lacks production-grade features like security or scalability.

You now have a fully functional, local instance of GPT-oSS running via Ollama, ready for chat, API use, and integration into more complex agentic applications.